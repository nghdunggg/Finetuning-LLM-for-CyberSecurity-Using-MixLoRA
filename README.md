Fine-tuning Mistral-7B-Instruct-v0.3 with QLoRA
Đây là dự án fine-tuning mô hình ngôn ngữ lớn Mistral-7B-Instruct-v0.3 sử dụng kỹ thuật QLoRA (Quantized Low-Rank Adaptation). QLoRA cho phép huấn luyện mô hình 7B một cách hiệu quả trên các GPU có bộ nhớ VRAM hạn chế (ví dụ: NVIDIA T4) bằng cách sử dụng lượng tử hóa 4-bit.

Mục tiêu của dự án là huấn luyện mô hình cơ sở với tập dữ liệu tùy chỉnh (normalized_merged_data.jsonl) cho các tác vụ chuyên biệt, sau đó hợp nhất (merge) adapter LoRA với mô hình gốc để tạo ra một mô hình độc lập, sẵn sàng cho việc chuyển đổi sang GGUF hoặc suy luận.

💻 1. Cài đặt và Khởi tạo Môi trường
Cài đặt Dependencies
Sử dụng file requirements.txt đi kèm để cài đặt tất cả các thư viện cần thiết:

pip install -r requirements.txt

Đăng nhập Hugging Face
Cần đăng nhập vào Hugging Face Hub để tải mô hình và báo cáo log huấn luyện (nếu sử dụng WandB):

from huggingface_hub import login
# Thay "HF_TOKEN" bằng token truy cập cá nhân của bạn
login("HF_TOKEN")

# Thiết lập thư mục cache tùy chỉnh (nếu cần)
import os
os.environ["HF_HUB_CACHE"] = "/data/llm_cyber/hf_cache_2"

⚙️ 2. Cấu hình Huấn luyện (QLoRA & SFT)
Model và Quantization
Mô hình được tải với cấu hình lượng tử hóa 4-bit (BitsAndBytesConfig) và sử dụng bfloat16 cho tính toán:

Cấu hình

Mô tả

Base Model

mistralai/Mistral-7B-Instruct-v0.3

Quantization

4-bit (QLoRA)

Compute Dtype

torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
)

LoRA Configuration (LoraConfig)
Cấu hình LoRA được áp dụng cho tất cả các module tự chú ý và feed-forward của mô hình Mistral để tối đa hóa khả năng học:

Tham số

Giá trị

Mục đích

r

8

Rank của ma trận LoRA.

lora_alpha

16

Hệ số tỷ lệ α.

target_modules

["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

Các module được LoRA áp dụng.

lora_dropout

0.05

Dropout rate.

Dữ liệu và Định dạng
Dữ liệu được tải từ file JSON Lines cục bộ (/data/llm_cyber/normalized_merged_data.jsonl) và định dạng theo tiêu chuẩn chat của Mistral:

# Tải dữ liệu từ local
dataset = load_dataset("json", data_files="/data/llm_cyber/normalized_merged_data.jsonl", split="train")

# Định dạng dữ liệu cho Mistral:
# <s>[INST] {instruction} [/INST] {response}</s>
def format_data_for_mistral(example):
    text = f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"
    return {"text": text}

Training Arguments (TrainingArguments)
Tham số

Giá trị

Mô tả

output_dir

./mistral-mixlora-finetune

Thư mục lưu checkpoint.

per_device_train_batch_size

2

Batch size trên mỗi thiết bị.

gradient_accumulation_steps

8

Tăng batch size hiệu quả lên 16.

learning_rate

2e-2

Tốc độ học.

max_steps

1000

Số bước huấn luyện tối đa.

gradient_checkpointing

True

Kỹ thuật tiết kiệm VRAM.

report_to

"wandb"

Báo cáo log lên Weights & Biases.

💾 3. Hợp nhất LoRA Adapter (Merge)
Sau khi hoàn tất fine-tuning, adapter LoRA được hợp nhất với mô hình cơ sở để tạo thành một mô hình FP16 đầy đủ, sẵn sàng cho suy luận.

Lưu ý quan trọng: Quá trình hợp nhất được thực hiện hoàn toàn trên CPU/RAM (device_map="cpu") do mô hình 7B (FP16) yêu cầu bộ nhớ lớn (khoảng 15GB).

Cấu hình Merge
Tham số

Giá trị

Mô tả

BASE_MODEL_ID

"mistralai/Mistral-7B-Instruct-v0.3"

Mô hình gốc.

LORA_ADAPTER_PATH

(Cập nhật đường dẫn của bạn)

Đường dẫn tới checkpoint LoRA đã huấn luyện (ví dụ: checkpoint-300).

OUTPUT_MERGED_MODEL_DIR

/data/llm_cyber/mistral_merged_model_gguf_ready

Thư mục lưu mô hình FP16 đã hợp nhất.

Code Snippet (Merge)
# 1. Tải Base Model vào CPU RAM (torch_dtype=torch.float16, device_map="cpu")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch # Đảm bảo đã import torch

# Xóa cache VRAM trước khi tải mô hình cơ sở lớn
if torch.cuda.is_available():
    torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",          
    low_cpu_mem_usage=True,
    # ... các tham số khác
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# 2. Tải LoRA Adapter
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# 3. Hợp nhất (Merge)
model = model.merge_and_unload()

# 4. Lưu mô hình đã hợp nhất (FP16)
model.save_pretrained(OUTPUT_MERGED_MODEL_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_MERGED_MODEL_DIR)
