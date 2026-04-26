ข้อ 1: จะทำอะไร

พัฒนาระบบ Pipeline Application Service แปลเสียงไทยเป็นอังกฤษแบบ Real-Time (Real-Time Thai Voice to English TTS Pipeline Deployed Service) โดยโฟกัสไปที่ฝั่ง Application (ทำเป็น Full Pipeline Service) แทนที่จะทำแค่ Research โมเดลอย่างเดียว Process หลักคือการเอาโมเดล 3 ส่วน (ASR, NMT, TTS) มา Integrate เข้าด้วยกัน แล้ว Optimize ความเร็วด้วยเทคนิค Quantization (เช่น INT8) พร้อมกับ Convert โมเดลให้อยู่ในฟอร์แมต Compute Graph (เช่น CTranslate2 หรือ ONNX) เพื่อให้ระบบรัน Inference ได้เร็วขึ้นและกิน Resource น้อยลง จนสามารถ Deploy ใช้งานได้จริงแบบ Real-Time 
---
ข้อ 2: Data อะไร (ห้ามบอกว่าจะเก็บเอง ต้องมี Data อยู่แล้ว)

ใช้ open source datasets ที่มีอยู่แล้ว โดยแบ่งการใช้งานตามแต่ละส่วนของ Pipeline ดังนี้:

Thai Audio (Module ASR):
https://huggingface.co/datasets/google/fleurs

Machine Translation (Translate Thai Text To English Text): https://huggingface.co/datasets/Salesforce/wikitext

English TTS (Module TTS): 
https://huggingface.co/datasets/pythainlp/scb_mt_enth_2020

---
. 
google/fleurs · Datasets at Hugging Face
We’re on a journey to advance and democratize artificial intelligence through open source and open science.
google/fleurs · Datasets at Hugging Face
Salesforce/wikitext · Datasets at Hugging Face
We’re on a journey to advance and democratize artificial intelligence through open source and open science.
Salesforce/wikitext · Datasets at Hugging Face
pythainlp/scb_mt_enth_2020 · Datasets at Hugging Face
We’re on a journey to advance and democratize artificial intelligence through open source and open science.
pythainlp/scb_mt_enth_2020 · Datasets at Hugging Face
ข้อ 3: โจทย์นี้วัดผลอย่างไร ขอลิงก์ paper, blog, github อย่างน้อย 2 อัน (ต้องมี paper 1 อัน) พร้อมบอกกระบวนการคร่าวๆ

การวัดผล (Evaluation):
จะวัดผลทั้งด้าน Speed และ Quality ในแต่ละ Module แยกกันและแบบ End-to-End รวมกัน ได้แก่:

Speed: วัดด้วยค่า Real-Time Factor (RTF) (ควรน้อยกว่า 1.0 เพื่อให้รันได้เร็วกว่าเวลาจริง) และค่า Latency ของ Pipeline รวม (End-to-End Latency)

Quality: วัดความแม่นยำของ ASR (โมดูลเสียงไทย) ด้วย Character Error Rate (CER) และวัด Quality การแปลภาษาของ MT ด้วยคะแนน BLEU หรือ COMET

Reference ที่ทำโจทย์แนวเดียวกัน:

Research Paper: "Speech-to-Speech Translation Pipelines for Conversations in Low-Resource Languages" (Published MT Summit 2025)
Link: https://aclanthology.org/2025.mtsummit-2.3.pdf
Process คร่าวๆ: Paper ทำ Cascaded Pipeline (เอา ASR -> MT -> TTS มาต่อกัน)  แทนที่จะใช้ End-to-End Model ตัวเดียว โดยเอา Pre-trained Model ที่มีอยู่แล้ว มาใช้ในกลุ่ม Low-Resource Language งานวิจัยเจอ Insight ว่า Performance Rank ของแต่ละ Module Independent กัน แปลว่าเราสามารถ Pick โมเดล ASR ตัวที่ Best ที่สุดมาต่อกับโมเดล MT ตัวที่ Best ที่สุดได้เลย

2: GitHub Repository: "speech-to-speech" (Hugging Face)
  
Link: https://github.com/huggingface/speech-to-speech
Process คร่าวๆ: เป็น Open-source Project ที่ทำ Modular Pipeline สำหรับระบบ Speech-to-Speech โดยเฉพาะ Process คือการรับเสียงเข้ามา ตัดช่วงเสียงด้วย VAD (Voice Activity Detection) ส่งเข้า STT (Speech-to-Text) โยนเข้า LLM/MT แล้วค่อยจบที่ TTS (Text-to-Speech) แบบ Local
 
