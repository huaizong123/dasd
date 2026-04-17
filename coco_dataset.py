import os
import json
import logging
import torch
from PIL import Image
from pycocotools.coco import COCO

from .base_dataset import BaseDataset 

logger = logging.getLogger(__name__)

class CoCoDataset(BaseDataset):
    def __init__(
        self,
        dataset_root: str,
        modalities: list,
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = True,
        tokenizeText: bool = False,
        split_prefix: str = "SpokenCOCO",
        **kwargs,
    ):
        # 1. 初始化 BaseDataset
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            image_transform=image_transform,
            audio_transform=audio_transform,
            target_sr=target_sr,
            load_audio=load_audio,
            load_image=load_image,
            tokenizeText=tokenizeText,
            **kwargs,
        )

        assert len(modalities) > 0, "Dataset's modalities cannot be empty"
        self.modalities = modalities
        
        # 🟢 您的安全檢查：確保 split 輸入正確
        assert self.split in ["train", "val", "test"], f"Invalid split: {self.split}"

        # 2. 載入 SpokenCOCO (找出圖片與聲音的對應)
        data_json_path = os.path.join(
            self.dataset_root, f"{split_prefix}_{self.split}.json"
        )
        logger.info(f"Loading SpokenCOCO from {data_json_path}")
        with open(data_json_path, "r") as f:
            raw_data = json.load(f)["data"]

        for _entry in raw_data:
            data_id = (
                _entry["reassign_id"]
                if split_prefix != "SpokenCOCO"
                else int(_entry["image"].split("_")[-1].replace(".jpg", ""))
            )
            for _caption in _entry["captions"]:
                _ent_data = {"id": data_id}
                
                # 🛡️ 讓它同時相容 "audio" 或 "wav"，避免在其他地方呼叫時出錯
                if "audio" in self.modalities or "wav" in self.modalities:
                    clean_wav_path = _caption["wav"].replace("wavs/", "")
                    _ent_data["wav"] = os.path.join(self.dataset_root, "SpokenCOCO", clean_wav_path)
                    
                if "image" in self.modalities:
                    # 原本有 "mscoco_img"，現在圖片(train2014/val2014)直接在根目錄下！
                    _ent_data["image"] = os.path.join(self.dataset_root, _entry["image"])
                    
                if "text" in self.modalities:
                    _ent_data["text"] = _caption["text"].lower()
                self.data.append(_ent_data)

        # ==========================================
        # 🟢 修改 3：MS COCO 標註檔也在根目錄，拔掉 "annotations"
        # ==========================================
        if self.split == "train":
            coco_ann_file = os.path.join(self.dataset_root, "instances_train2014.json")
        else:
            coco_ann_file = os.path.join(self.dataset_root, "instances_val2014.json")
            
        logger.info(f"Loading COCO Annotations from {coco_ann_file}")
        self.coco = COCO(coco_ann_file)

    def __getitem__(self, index):
        entry = self.data[index]
        img_id = entry["id"]
        
        ret_dict = {"id": img_id}

        # --- 處理聽覺 ---
        if "wav" in entry and self.load_audio:
            ret_dict["wav"] = self._LoadAudio(entry["wav"])
            
        # --- 處理文字 ---
        if "text" in entry:
            if self.tokenizeText:
                ret_dict["text"] = self._TokenizeText(entry["text"])
            else:
                # 🌟 補上這行：如果不需要提早切詞，就直接回傳原始字串給 GroundingDINO！
                ret_dict["text"] = entry["text"]

        # --- 處理視覺與 Bounding Box ---
        if "image" in entry and self.load_image:
            img_path = entry["image"]
            image = Image.open(img_path).convert("RGB")
            w, h = image.size 

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            boxes, labels = [], []
            for ann in anns:
                # 只保留有效的 Box
                if ann.get('ignore', False) or ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                    continue
                
                x_min, y_min, box_w, box_h = ann['bbox']

                # 🌟 致命修正 1：只要算出絕對像素的 [x_min, y_min, x_max, y_max] 就好！
                # 千萬不要在這裡除以寬高，也不要轉成中心點 cxcy！
                x_max = x_min + box_w
                y_max = y_min + box_h
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(0)

            if len(boxes) > 0:
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                    "orig_size": torch.as_tensor([int(h), int(w)]),
                    "size": torch.as_tensor([int(h), int(w)]) # 🌟 建議補上 size，有些 Transform 會看這個
                }
            else:
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "orig_size": torch.as_tensor([int(h), int(w)]),
                    "size": torch.as_tensor([int(h), int(w)])
                }

            # 🌟 致命修正 2：必須同時把 image 和 target 傳給 transform！
            if self.image_transform is not None:
                image, target = self.image_transform(image, target)

            ret_dict["image"] = image
            ret_dict["target"] = target 

        return ret_dict