import os
import sys
sys.path.append('/workspace/data/SpeechCLIP')
import argparse
import yaml
import logging
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# PyTorch 效能與除錯設定
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
#torch.backends.cuda.enable_flash_sdp(False)
#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_math_sdp(True)

# ===== 匯入您的自定義模組 =====
from avssl.data.coco_dataset import CoCoDataset
from avssl.data.collate_function import collate_general
from avssl.module.losses import SpeechDINOLoss
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.datasets import transforms as T
from vis_utils import visualize_debug

# ===== 匯入 SpeechCLIP 提供的四大神器 =====
from avssl.util.model_utils import freeze_model, unfreeze_model
from avssl.util.data_utils import get_keypadding_mask
from avssl.util.init_model import init_weights
from avssl.util.penalty_scheduler import PenaltyScheduler



def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SpeechDINO Training", add_help=False)

    # 1. Config & ckpt (雙 Config 設計)
    parser.add_argument("--gdino_config", type=str, required=True, help="GroundingDINO 的 .py 設定檔")
    parser.add_argument("--sclip_config", type=str, required=True, help="SpeechCLIP 的 .yaml 設定檔")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Directory to save ckpts.")
    parser.add_argument("--dataset_root", type=str, default="/workspace/data/SpeechCLIP", help="Override dataset root")

    # 2. Mode (控制程式行為)
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--eval", action="store_true", help="Run dev set")
    parser.add_argument("--test", action="store_true", help="Run test set")
    parser.add_argument("--ckpt", type=str, default="", help="Load from checkpoint (for eval/test)")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume training.")
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    # 3. Hparams
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--njobs", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--seed", type=int, default=7122, help="Random seed")
    
    return parser

def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize([400, 480, 512, 576, 600], max_size=800),
                T.Compose([
                    T.RandomResize([400, 500]),
                    T.RandomSizeCrop(384, 500),
                    T.RandomResize([400, 480, 512, 576, 600], max_size=800),
                ])
            ),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([512], max_size=900),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')

def main():
    parser = argparse.ArgumentParser("SpeechDINO", parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(args.save_path, 'training_log.txt')), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔥 使用設備: {device}")

    # --- Config Setup ---
    cfg = SLConfig.fromfile(args.gdino_config)
    with open(args.sclip_config, 'r') as f:
        cfg.speechclip_args = yaml.safe_load(f)
    cfg.output_dir = args.save_path

    # ==========================================
    # 1. 資料集初始化
    # ==========================================
    logger.info("📦 正在載入 Dataset (啟用多尺度 BBox 增強)...")
    train_dataset = CoCoDataset(
        dataset_root=args.dataset_root,
        modalities=["image", "audio", "text"],
        split="train",
        load_audio=True, load_image=True, tokenizeText=False,
        image_transform=make_coco_transforms('train')
    )
    val_dataset = CoCoDataset(
        dataset_root=args.dataset_root,
        modalities=["image", "audio"],
        split="val",
        load_audio=True, load_image=True, tokenizeText=False,
        image_transform=make_coco_transforms('val')
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_general, num_workers=args.njobs, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_general, num_workers=args.njobs, pin_memory=True)

    # ==========================================
    # 🧠 2. 模型建立與三階段權重注入
    # ==========================================
    logger.info("🧠 建立模型與載入靈魂...")
    model = build_model(cfg)

    # A. 視覺大腦 (GroundingDINO)
    gdino_ckpt = torch.load("/workspace/data/groundingdino_swinb_cogcoor.pth", map_location="cpu")
    model_dict = model.state_dict()
    matched_dict = {k: v for k, v in gdino_ckpt.get('model', gdino_ckpt).items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)

    # B. 聽覺大腦 (SpeechCLIP HuBERT)
    sclip_ckpt = torch.load("/workspace/data/SpeechCLIP/slt_ckpts/SpeechCLIP/base/flickr/parallel/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt", map_location="cpu", weights_only=False)
    sc_state_dict = sclip_ckpt.get('state_dict', sclip_ckpt.get('model_state_dict', sclip_ckpt))
    audio_weights = {}
    for k, v in sc_state_dict.items():
        if 'weightedsum_layer' in k: audio_weights['speech_encoder.weightedsum_layer.weight'] = v
        elif 'audio_encoder' in k or 'speech_encoder' in k:
            name_part = k.split('audio_encoder.')[-1] if 'audio_encoder' in k else k.split('speech_encoder.')[-1]
            audio_weights[f'speech_encoder.{name_part}'] = v
    model.load_state_dict(audio_weights, strict=False)

    # C. 語意靈魂 (Q-Former Epoch 5)
    qformer_ckpt = torch.load("ckp/keyword_prober/qformer_epoch_5.pth", map_location="cpu")
    mapped_qformer = {k.replace("qformer.", "audio_qformer."): v for k, v in qformer_ckpt.items() if k.startswith("qformer.")}
    model.load_state_dict(mapped_qformer, strict=False)

    logger.info("✅ 視覺、聽覺、語意 三大權重注入完成！")

    # ==========================================
    # ❄️ 3. Stage 2 專屬凍結策略 (乾淨俐落)
    # ==========================================
    trainable_count = 0
    for name, param in model.named_parameters():
        # 絕對凍結區：Swin 骨幹與 HuBERT 骨幹
        if "backbone" in name or "speech_encoder" in name:
            param.requires_grad = False
        # 解凍區：翻譯官(QFormer/Proj) + 融合大腦(Transformer) + 預測頭(Embed)
        elif any(x in name for x in ['audio_qformer', 'speech_proj', 'transformer', 'bbox_embed', 'class_embed']):
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            
    logger.info(f"❄️ 凍結完成！共有 {trainable_count} 個模組張量準備進行畫框微調。")
    model = model.to(device)

    # ==========================================
    # 🎯 4. Loss, Optimizer, Scheduler
    # ==========================================
    criterion = SpeechDINOLoss(
        stage=2, # 強制 Stage 2
        num_classes=1, loss_weight_class=cfg.dn_label_coef, loss_weight_bbox=cfg.dn_bbox_coef, loss_weight_giou=2.0
    ).to(device)

    base_lr = 1e-4  
    
    # 🌟 互斥分組法：確保沒有參數重複！
    qformer_params = []
    proj_params = []
    embed_params = []
    transformer_params = []
    other_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        # ⚠️ 順序很重要：先抓最細的，再抓大範圍的！
        if "audio_qformer" in n:
            qformer_params.append(p)
        elif "speech_proj" in n:
            proj_params.append(p)
        elif "class_embed" in n or "bbox_embed" in n:
            embed_params.append(p) # 這裡會把 transformer 裡面的 embed 也抓走
        elif "transformer" in n:
            transformer_params.append(p) # 剩下的 transformer 才會來這裡
        else:
            other_params.append(p) # 漏網之魚 (如 class_bias) 放這裡

    # 建立不會打架的群組
    param_dicts = [
        {"params": qformer_params, "lr": base_lr * 0.1}, # 慢速保護
        {"params": proj_params + other_params, "lr": base_lr * 1.0}, # 投影層與雜項
        {"params": embed_params, "lr": base_lr * 4.0}, # 極速開竅 (預測頭)
        {"params": transformer_params, "lr": base_lr * 0.5} # Transformer 融合大腦
    ]
    
    # 別忘了把 Criterion (Loss) 裡面需要學習的參數也加進來 (例如溫度係數)
    for n, p in criterion.named_parameters():
        if p.requires_grad: 
            param_dicts[1]["params"].append(p)

    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)
    
    num_epochs = 10
    total_steps = len(train_loader) * num_epochs 
    warmup_steps = 2000 # Stage 2 暖機短一點
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: float(s)/warmup_steps if s < warmup_steps else max(0.0, float(total_steps - s)/(total_steps - warmup_steps)))

    # ... 前面的參數解析與模型初始化 ...
    
    # --- Resume Logic ---
    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f"🔄 從 Epoch {start_epoch+1} 恢復訓練！")

    # ==========================================
    # 🌟 攔截點：如果設定了 --eval，只跑 Validation 就結束
    # ==========================================
    if args.eval:  # 假設你的 argparse 參數名稱叫 eval
        logger.info("🚀 [驗證模式] 強制進入 Validation 測試 Checkpoint...")
        
        model.eval()
        val_loss = 0.0
        
        # 如果你沒有用到 criterion_loss，可以忽略。
        # 為了保險，我們加上 try-except 捕捉任何異常
        try:
            with torch.no_grad():
                # 為了分辨這是 Eval 模式，我們把進度條名稱改一下
                for val_batch in tqdm(val_loader, desc="Testing Model (Eval Only)"):
                    v_img = val_batch["image"].to(device)
                    v_wavs = val_batch["wav"].to(device)
                    v_wav_lens = torch.as_tensor(val_batch["wav_len"]).to(device)
                    
                    # 防禦 0：圖片太大直接跳過
                    if hasattr(v_img, 'tensors') and (v_img.tensors.shape[2] > 1200 or v_img.tensors.shape[3] > 1200): 
                        continue
                    
                    # 🛡️ 終極音長防護網 (Val)
                    # 防禦 1：攔截極短或損壞的音檔
                    if v_wav_lens.min() < 400:
                        continue

                    # 防禦 2：超長音檔截斷
                    MAX_WAV = 480000
                    if v_wavs.size(1) > MAX_WAV: 
                        v_wavs = v_wavs[:, :MAX_WAV]
                        v_wav_lens = torch.clamp(v_wav_lens, max=MAX_WAV)

                    v_mask = get_keypadding_mask(v_wavs.size(1), v_wav_lens).to(device)
                    
                    v_targets = [{k: v.to(device) for k, v in t.items()} for t in val_batch["target"]]
                    for t in v_targets:
                        if "labels" in t: t["labels"] = torch.zeros_like(t["labels"])

                    v_out = model(samples=v_img, wavs=v_wavs, audio_mask=v_mask, wav_lens=v_wav_lens)
                    
                    if torch.isnan(v_out['pred_boxes']).any(): 
                        continue
                    
                    vl, _ = criterion(v_out, v_targets)
                    if not torch.isnan(vl): 
                        val_loss += vl.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"✅ [測試完成] Validation Loss: {avg_val_loss:.4f}")
            
        except Exception as e:
            logger.error(f"❌ 驗證過程發生錯誤: {e}")
            import traceback
            traceback.print_exc()

        logger.info("程式結束 (Eval Only Mode)。")
        import sys
        sys.exit(0) # 👈 這行最重要！跑完 Validation 直接強制終止，絕對不往下執行 Train
    # ==========================================

    # 原本的訓練迴圈 (因為有 sys.exit，如果是 --eval 就不會走到這裡)
    #for epoch in range(start_epoch, max_epochs):
    #    pass
    #    # train(...)
    
    # ==========================================
    # 🏃 5. 訓練主迴圈
    # ==========================================
    best_val_loss = float('inf')
    vis_dir = os.path.join(cfg.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        criterion.train()
        # 強制 Backbone 進入 eval 模式 (關閉 BatchNorm 和 Dropout)
        if hasattr(model, 'backbone'): model.backbone.eval()
        if hasattr(model, 'speech_encoder'): model.speech_encoder.eval()

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train E{epoch+1}")
        accumulation_steps = 4
        optimizer.zero_grad()
        consecutive_nan = 0

        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            wavs = batch["wav"].to(device)
            wav_lens = torch.as_tensor(batch["wav_len"]).to(device)

            # 防禦 1：攔截極短或損壞的音檔 (避免 CNN 卷積算出負數)
            if wav_lens.min() < 400:
                # logger.warning(f"⚠️ 攔截到極短音檔 (長度 {wav_lens.min()})，跳過此 Batch！")
                optimizer.zero_grad()
                continue

            # 防禦 2：超長音檔截斷，必須「波形與長度紀錄」同步截斷！
            MAX_WAV = 128000
            if wavs.size(1) > MAX_WAV:
                wavs = wavs[:, :MAX_WAV]
                # 🌟 致命修正：把長度紀錄也強壓在 MAX_WAV 以內！
                wav_lens = torch.clamp(wav_lens, max=MAX_WAV)

            audio_mask = get_keypadding_mask(wavs.size(1), wav_lens).to(device)

            # 目標格式轉換 (配合 DINO Criterion)
            targets = []
            for t in batch["target"]:
                t_dict = {k: v.to(device) for k, v in t.items()}
                if "labels" in t_dict: t_dict["labels"] = torch.zeros_like(t_dict["labels"])
                targets.append(t_dict)

            # --- Forward ---
            outputs = model(samples=images, wavs=wavs, audio_mask=audio_mask, wav_lens=wav_lens)
            
            if torch.isnan(outputs['pred_boxes']).any() or torch.isnan(outputs['pred_logits']).any():
                optimizer.zero_grad()
                continue

            loss, log_dict = criterion(outputs, targets)
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 2000:
                optimizer.zero_grad()
                continue

            # --- Backward & Step ---
            current_loss = loss.item()
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), 0.5)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    consecutive_nan += 1
                    if consecutive_nan >= 10: exit(1)
                    optimizer.zero_grad()
                    continue
                consecutive_nan = 0
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += current_loss
            global_step += 1

            # --- Logging ---
            progress_bar.set_postfix({"Total": f"{current_loss:.4f}", "BBox": f"{log_dict.get('loss_bbox', 0):.4f}"})
            if (global_step + 1) % 5000 == 0:
                latest_ckpt_path = os.path.join(cfg.output_dir, "speechdino_latest.pth")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, latest_ckpt_path)
                
                # 順便存一個帶有 step 數字的版本，避免被覆蓋
                step_ckpt_path = os.path.join(cfg.output_dir, f"speechdino_step_{global_step+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, step_ckpt_path)
                
                logger.info(f"💾 [微步存檔] 已儲存 Step {global_step+1} 的進度至 {step_ckpt_path}")

            # --- Visualization ---
            if (global_step + 1) % 500 == 0:
                model.eval()
                with torch.no_grad():
                    v_img = images.tensors[0] if hasattr(images, 'tensors') else images[0]
                    visualize_debug(
                        image_tensor=v_img, image_dense_feat=outputs['image_dense_feat'][0],
                        audio_cls_feat=outputs['audio_tokens'][0].mean(dim=0), # 畫圖用平均特徵即可
                        pred_boxes=outputs['pred_boxes'][0], pred_logits=outputs['pred_logits'][0],
                        output_path=vis_dir, step=global_step+1
                    )
                model.train()
                if hasattr(model, 'backbone'): model.backbone.eval()
                if hasattr(model, 'speech_encoder'): model.speech_encoder.eval()

        # ==========================================
        # 🔍 Validation & Checkpointing
        # ==========================================
        avg_train_loss = epoch_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f"Val E{epoch+1}"):
                v_img = val_batch["image"].to(device)
                v_wavs = val_batch["wav"].to(device)
                v_wav_lens = torch.as_tensor(val_batch["wav_len"]).to(device)
                
                # 防禦 0：圖片太大直接跳過
                if hasattr(v_img, 'tensors') and (v_img.tensors.shape[2] > 1200 or v_img.tensors.shape[3] > 1200): 
                    continue
                
                # ==========================================
                # 🛡️ 終極音長防護網 (Val)
                # ==========================================
                # 防禦 1：攔截極短或損壞的音檔
                if v_wav_lens.min() < 400:
                    continue

                # 防禦 2：超長音檔截斷 (波形與長度紀錄同步)
                MAX_WAV = 128000
                if v_wavs.size(1) > MAX_WAV: 
                    v_wavs = v_wavs[:, :MAX_WAV]
                    # 🌟 致命修正：把長度紀錄也強壓在 MAX_WAV 以內！
                    v_wav_lens = torch.clamp(v_wav_lens, max=MAX_WAV)

                v_mask = get_keypadding_mask(v_wavs.size(1), v_wav_lens).to(device)
                # ==========================================
                
                v_targets = [{k: v.to(device) for k, v in t.items()} for t in val_batch["target"]]
                for t in v_targets:
                    if "labels" in t: t["labels"] = torch.zeros_like(t["labels"])

                v_out = model(samples=v_img, wavs=v_wavs, audio_mask=v_mask, wav_lens=v_wav_lens)
                if torch.isnan(v_out['pred_boxes']).any(): 
                    continue
                
                # ==========================================
                # 🛡️ 新增防禦：過濾 Inf 並限制座標範圍
                # ==========================================
                # 1. 擋下無限大 (Infinity) 的異常預測值
                if torch.isinf(v_out['pred_boxes']).any():
                    continue
                    
                # 2. 強制把預測框的座標限制在 [0, 1] 之間，防止越界計算 GIoU 時炸毀記憶體
                v_out['pred_boxes'] = torch.clamp(v_out['pred_boxes'], min=0.0, max=1.0)
                
                # 3. 防禦空張量：確保 target 裡面真的有 box 可以算 loss
                has_empty_target = False
                for t in v_targets:
                    if 'boxes' not in t or len(t['boxes']) == 0:
                        has_empty_target = True
                        break
                if has_empty_target:
                    continue
                # ==========================================
                
                vl, _ = criterion(v_out, v_targets)
                if not torch.isnan(vl): val_loss += vl.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"📊 Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoint
        ckpt_data = {
            'epoch': epoch + 1, 'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(ckpt_data, os.path.join(cfg.output_dir, f"speechdino_epoch_{epoch+1}.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_data, os.path.join(cfg.output_dir, "speechdino_best.pth"))
            logger.info("🌟 更新 Best Checkpoint！")

if __name__ == "__main__":
    main()