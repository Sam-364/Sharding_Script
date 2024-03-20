import torch
from tranformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, load_checkpoint_and_dispatch
model_name = "Nexusflow/NexusRaven-V2-13B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
accelerator = Accelerator()
save_directory = "/media/test-vm-gpu/US_Storage/NexusRaven V-2_Quantized"
accelerator.save_model(
    model=model,
    save_directory=save_directory,
    max_shard_size="200MB"	
)
device_map = {"": 'cpu'}
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=save_directory,
    device_map=device_map,
    no_split_module_classes=[]
)
new_model = "NexusRaven_v2_sharded"
HF_TOKEN = "HF_API_TOKEN"
tokenizer.push_to_hub(
    new_model,
    token=HF_TOKEN,
)
