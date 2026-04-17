import os
import sys
sys.path.append('/workspace/data/SpeechCLIP')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# 匯入你專案中的模組
from avssl.data.coco_dataset import CoCoDataset
from avssl.data.collate_function import collate_general
from avssl.module.speech_encoder_plus import FairseqSpeechEncoder_Hubert
from avssl.util.data_utils import get_keypadding_mask

# ==========================================
# 1. 標籤產生器 (COCO 80 類)
# ==========================================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def create_keyword_labels(texts, device):
    """將文字轉換為 80 維的 0/1 多標籤矩陣"""
    batch_size = len(texts)
    labels = torch.zeros((batch_size, len(COCO_CLASSES)), dtype=torch.float32, device=device)
    
    for i, text in enumerate(texts):
        # 加上空格避免子字串誤判 (例如 "carpet" 被判成 "car")
        text_padded = f" {text.lower()} " 
        for j, keyword in enumerate(COCO_CLASSES):
            if f" {keyword} " in text_padded:
                labels[i, j] = 1.0
    return labels

# ==========================================
# 2. 模型定義：Q-Former768 + 分類器
# ==========================================
class AudioQFormer768(nn.Module):
    def __init__(self, num_queries=32, nheads=12, num_layers=2):
        super().__init__()
        self.hidden_dim = 768
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, self.hidden_dim))
        nn.init.normal_(self.query_embed, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=nheads, dim_feedforward=self.hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, audio_feat, audio_pad_mask):
        bs = audio_feat.shape[0]
        queries = self.query_embed.expand(bs, -1, -1)
        return self.transformer(tgt=queries, memory=audio_feat, memory_key_padding_mask=audio_pad_mask)

class KeywordBottleneckModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qformer = AudioQFormer768()
        # 關鍵：接上一個 Linear 層，把 768 維對應到 80 個類別
        self.classifier = nn.Linear(768, len(COCO_CLASSES))

    def forward(self, audio_feat, audio_mask):
        # 1. 壓縮成 32 個 Tokens
        tokens = self.qformer(audio_feat, audio_mask) # (B, 32, 768)
        # 2. 取平均得到整句話的語意
        sentence_emb = tokens.mean(dim=1) # (B, 768)
        # 3. 預測 80 個類別
        logits = self.classifier(sentence_emb) # (B, 80)
        return logits

# ==========================================
# 3. 訓練主程式
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動語意瓶頸訓練 (設備: {device})")

    # --- A. 準備資料 ---
    dataset_root = "/workspace/data/SpeechCLIP/data" # 換成你真實的路徑
    print("📦 正在載入 Dataset...")
    train_dataset = CoCoDataset(
        dataset_root=dataset_root,
        modalities=["audio", "text"], # 只要聲音跟文字，拔掉影像加速
        split="train",
        load_audio=True, load_image=False, tokenizeText=False # 👈 關鍵：不切詞
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, # 只有聲音，Batch Size 可以開大
        collate_fn=collate_general, num_workers=4, pin_memory=True
    )

    # --- B. 準備模型 ---
    print("🧠 正在初始化 HuBERT 與 Q-Former 分類器...")
    speech_encoder = FairseqSpeechEncoder_Hubert(
        name="hubert_base", 
        pretrained=False, 
        trainable=False,
        feat_select_idx="weighted_sum", 
        normalize_hiddenstates=True, 
        normalize_type="s3prl"
    ).to(device)
    
    # 【重要】載入 HuBERT 預訓練權重 (請確認你的路徑)
    speechclip_ckpt = "/workspace/data/SpeechCLIP/slt_ckpts/SpeechCLIP/base/flickr/parallel/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt"
    if os.path.exists(speechclip_ckpt):
        ckpt = torch.load(speechclip_ckpt, map_location="cpu", weights_only=False)
        sc_state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        audio_weights = {k.split('audio_encoder.')[-1] if 'audio_encoder' in k else k.split('speech_encoder.')[-1]: v for k, v in sc_state_dict.items() if 'audio_encoder' in k or 'speech_encoder' in k}
        if 'weightedsum_layer' in sc_state_dict:
            audio_weights['weightedsum_layer.weight'] = sc_state_dict['weightedsum_layer']
        speech_encoder.load_state_dict(audio_weights, strict=False)
        print("✅ HuBERT 聽力大腦載入完成並已凍結！")
    
    speech_encoder.eval() # 永遠凍結
    
    model = KeywordBottleneckModel().to(device)
    model.train()

    # --- C. 訓練設定 ---
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() # 處理多標籤的黃金 Loss

    # --- D. 訓練迴圈 ---
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\n========== Epoch [{epoch+1}/{num_epochs}] ==========")
        progress_bar = tqdm(train_loader, desc="Training")
        epoch_loss = 0.0
        
        for step, batch in enumerate(progress_bar):
            wavs = batch["wav"].to(device)
            texts = batch["text"] # List of strings
            wav_lens = torch.as_tensor(batch["wav_len"]).to(device)
            
            # ==========================================
            # 🛡️ 終極音長防禦網 (防 CUDA 崩潰)
            # ==========================================
            # 1. 防禦極短音檔
            if wav_lens.min() < 400:
                # 靜默跳過，避免洗版
                continue
                
            # 2. 防禦極長音檔 (同步截斷波形與長度紀錄)
            MAX_WAV_LENGTH = 480000
            if wavs.size(1) > MAX_WAV_LENGTH:
                wavs = wavs[:, :MAX_WAV_LENGTH]
                wav_lens = torch.clamp(wav_lens, max=MAX_WAV_LENGTH)
            # ==========================================
            
            # 3. 產生真實答案 (Ground Truth)
            targets = create_keyword_labels(texts, device)
            
            # 4. 提取 HuBERT 特徵，並取得「壓縮後的長度 (feat_lens)」
            with torch.no_grad():
                # 🌟 終極解法 1：強制關閉自動混合精度，使用最穩定的 FP32
                with torch.cuda.amp.autocast(enabled=False):
                    # 確保輸入是純淨的 float32
                    wavs_fp32 = wavs.float()
                    try:
                        encoded_audio, feat_lens = speech_encoder(wav=wavs_fp32, wav_len=wav_lens)
                    except Exception as e:
                        print(f"\n⚠️ [防護網] HuBERT 內部崩潰，已攔截！錯誤訊息: {e}")
                        continue # 直接跳過這批毒資料
                
            # 5. 確保輸出的 Tensor 裡面沒有 NaN 病毒
            if torch.isnan(encoded_audio).any() or torch.isinf(encoded_audio).any():
                print("\n⚠️ [防護網] 攔截到 NaN 病毒特徵，跳過此 Batch！")
                continue

            # 6. 使用「壓縮後的長度」製作 Mask
            audio_mask = get_keypadding_mask(encoded_audio.shape[1], feat_lens).to(device)
            
            # 7. 送入 Q-Former 預測 80 個類別
            # 🌟 終極解法 2：Q-Former 也強制用 FP32 算
            with torch.cuda.amp.autocast(enabled=False):
                logits = model(encoded_audio.float(), audio_mask)
            
            # 8. 計算 Loss 與更新
            loss = criterion(logits, targets)

            # 🛡️ 終極解法 3：反向傳播前檢查 Loss 是否正常
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                print(f"\n🚨 [防護網] 抓到異常 Loss ({loss.item()})，跳過更新！")
                optimizer.zero_grad()
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # 🎯 狙擊手開竅監控：每 200 步看看模型學到了什麼
            if step % 200 == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits[0]) # 拿 Batch 裡的第一句話來偷看
                    top_prob, top_idx = torch.max(probs, dim=0)
                    if top_prob > 0.3: # 有點自信才印出來
                        pred_word = COCO_CLASSES[top_idx.item()]
                        real_text = texts[0]
                        print(f"\n💡 [實況轉播] 聽到句子: '{real_text}'")
                        print(f"   => 模型大膽猜測包含: [{pred_word}] (信心度 {top_prob:.2f})")

        print(f"📈 Epoch {epoch+1} 結束 | 平均 Loss: {epoch_loss/len(train_loader):.4f}")
        
        # 存檔
        os.makedirs("ckp/keyword_prober", exist_ok=True)
        torch.save(model.state_dict(), f"ckp/keyword_prober/qformer_epoch_{epoch+1}.pth")
        print("💾 權重已儲存！")

if __name__ == "__main__":
    main()