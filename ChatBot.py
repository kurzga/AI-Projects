import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Örnek veri seti
data = [
    "Bu bir örnek cümle.",
    "Başka bir örnek cümle.",
    "Bu da bir cümle örneği.",
    "Örnekler devam ediyor.",
    "Son örnek cümle.",
    "nasılsın?",
    "aslında iyiyim.Sen?",
    "Ben de iyiyim."
]

# Veri setini tokenize etmek için sınıf tanımı
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        return torch.tensor(tokens, dtype=torch.long)

# Tokenizer ve modeli yükle
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config)

# Veri seti oluştur
max_length = 32
dataset = TextDataset(data, tokenizer, max_length)
# Verileri aynı uzunluğa sahip olacak şekilde dolgu yap
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: pad_sequence(x, batch_first=True))

# Modeli eğit
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch.to(model.device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Average Loss: {total_loss / len(dataloader)}")

# Kullanıcıdan girdi al
user_input = input("Soru: ")

# Modeli değerlendir ve yanıtı al
input_ids = tokenizer.encode(user_input, return_tensors='pt')
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Kullanıcıya yanıtı göster
print("ChatGPT: ", response)
