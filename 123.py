import torch
import torch.nn as nn
import math
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ==========================================
# 0. 定義特殊 Token 與參數
# ==========================================
PAD_IDX = 0
BOS_IDX = 1  # Begin of Sequence
EOS_IDX = 2  # End of Sequence
VOCAB_SIZE = 500  # 假設你提取出的名詞/動詞字典大小為 500
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 3

# ==========================================
# 1. 位置編碼 (Positional Encoding)
# ==========================================
# Transformer 必須要知道 Token 的順序
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (Batch, Seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 2. 核心模型：HuBERT + Projector + Transformer Decoder
# ==========================================
class AudioToSemanticModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers):
        super().__init__()
        
        # --- Encoder: 凍結的 HuBERT ---
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert.eval()
        for param in self.hubert.parameters():
            param.requires_grad = False
            
        hubert_dim = self.hubert.config.hidden_size # 768
        
        # --- Projector: 將 HuBERT 維度對齊到 Decoder 維度 ---
        self.projector = nn.Sequential(
            nn.Linear(hubert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # --- Decoder 元件 ---
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True # 確保輸入形狀為 (Batch, Seq, Feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # --- 輸出分類層 ---
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    # 產生 Causal Mask，防止 Decoder 偷看未來的字詞
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_values, tgt_tokens):
        """
        input_values: (Batch, Audio_Seq_Len) - 原始音訊
        tgt_tokens: (Batch, Tgt_Seq_Len) - 教師強制的輸入標籤 (BOS 開頭)
        """
        device = input_values.device
        
        # 1. 取得音訊特徵 (Memory)
        with torch.no_grad():
            hubert_out = self.hubert(input_values).last_hidden_state # (Batch, Time, 768)
            
        # 將音訊特徵投射到 hidden_dim，作為 Transformer 的 Memory (Key & Value)
        memory = self.projector(hubert_out) # (Batch, Time, hidden_dim)
        
        # 2. 處理文字特徵 (Query)
        tgt_emb = self.embedding(tgt_tokens) # (Batch, Tgt_Seq, hidden_dim)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # 3. 建立 Masks
        # 確保 Decoder 只能看前面的字 (Causal Mask)
        tgt_seq_len = tgt_tokens.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        
        # 確保 Decoder 忽略 PAD token (Padding Mask)
        tgt_key_padding_mask = (tgt_tokens == PAD_IDX)
        
        # 4. Transformer Decoder 前向傳播
        # memory: 音訊特徵 (Key, Value)
        # tgt_emb: 文字特徵 (Query)
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        ) # (Batch, Tgt_Seq, hidden_dim)
        
        # 5. 輸出預測
        logits = self.fc_out(output) # (Batch, Tgt_Seq, vocab_size)
        return logits

# ==========================================
# 3. 模擬訓練迴圈 (包含 Teacher Forcing)
# ==========================================
def train_pipeline():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用硬體: {DEVICE}")

    # 初始化模型
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = AudioToSemanticModel(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    
    # 只有 Projector, Embedding, Decoder 參與訓練
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # 忽略 PAD_IDX 的 Loss 計算
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # --- 模擬 Batch 資料 ---
    BATCH_SIZE = 2
    # 模擬音訊
    dummy_audio = [torch.randn(16000 * 3).numpy() for _ in range(BATCH_SIZE)]
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)

    # 模擬 目標語意序列 (Ground Truth)
    # 假設詞彙對應: 10=dog, 25=run, 30=grass
    # 第一句話正確序列: <BOS>, dog, run, grass, <EOS>, <PAD>
    # 第二句話正確序列: <BOS>, cat, sleep, <EOS>, <PAD>, <PAD>
    targets = torch.tensor([
        [BOS_IDX, 10, 25, 30, EOS_IDX, PAD_IDX], 
        [BOS_IDX, 15, 40, EOS_IDX, PAD_IDX, PAD_IDX]
    ]).to(DEVICE)

    print("\n開始模擬訓練...")
    model.train()
    model.hubert.eval() # 再次確保 HuBERT 處於 eval 狀態

    optimizer.zero_grad()
    
    # [關鍵技巧：Teacher Forcing]
    # Decoder 的輸入不需要包含最後一個 Token (EOS 後面那個)
    decoder_input = targets[:, :-1] 
    
    # 預測的目標答案不需要包含第一個 Token (BOS)
    expected_output = targets[:, 1:] 

    # 前向傳播
    logits = model(input_values, decoder_input)
    
    # 計算 Loss
    # logits shape 轉換: (Batch, Seq, Vocab) -> (Batch * Seq, Vocab)
    # expected_output shape 轉換: (Batch, Seq) -> (Batch * Seq)
    loss = criterion(logits.reshape(-1, VOCAB_SIZE), expected_output.reshape(-1))
    
    print(f"Step Loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    print("反向傳播與參數更新成功！")

if __name__ == "__main__":
    train_pipeline()