# Abstract

Khmer is under-resourced for automatic speech recognition (ASR), and published Khmer results cannot
be compared because they use different test sets, text normalization and training data. This thesis
presents a controlled study of fine-tuning pretrained speech models for Khmer. A full audit of the
largest open Khmer corpus (450,396 utterances, 1,073.9 hours, 20 speakers) shows that every sentence
is read by four speakers, so a random split leaks test sentences into training. We therefore build a
split that holds out both the test speakers and every sentence they read, a documented Khmer
normalization protocol, and a space-insensitive character error rate (CER).

Without fine-tuning, Whisper models fail on Khmer (CER above 100%), while MMS-1B with its Khmer
adapter reaches 9.71%. Fine-tuning on nested 10, 50 and 100-hour subsets improves both families:
Whisper small falls from 19.43% to 4.70% CER and reaches 2.85% with the full 872 hours, and MMS falls
from 8.43% to 5.24%. Starting the MMS adapter from the released Khmer adapter instead of from scratch
lowers CER at every data size (to 4.62% at 100 hours) and gives the most robust model, the only
fine-tuned model better than zero-shot on the out-of-domain FLEURS benchmark (14.36% against
15.49%). Whisper wins in domain but loses out of domain (18.61% on FLEURS), and it is 27–34% more
accurate on speakers seen in training than on new speakers, against 8–11% for MMS, so held-out
speakers are essential for a fair estimate. An error analysis shows that the most frequent errors of
every model are spelling variants that the corpus itself mixes, and that digits, English words and
long transcripts are the weakest points. All fine-tuning cost about 23 USD of GPU time; the code,
configurations, split manifests and run logs are public.

**Keywords:** Khmer, automatic speech recognition, fine-tuning, Whisper, MMS, low-resource languages,
evaluation protocol, character error rate

---

# សេចក្តីសង្ខេប (Khmer draft — must be reviewed by the author before use)

> Drafted by an AI assistant from the English abstract. Check the wording, the technical terms and the
> number style (Khmer or Arabic digits; decimal separator) against the RUPP template before use.

ភាសាខ្មែរនៅខ្វះធនធានសម្រាប់ការសម្គាល់សំឡេងនិយាយដោយស្វ័យប្រវត្តិ (ASR) ហើយលទ្ធផលដែលបានបោះពុម្ពផ្សាយ
សម្រាប់ភាសាខ្មែរ មិនអាចប្រៀបធៀបគ្នាបានទេ ដោយសារវាប្រើសំណុំទិន្នន័យតេស្ត វិធីធ្វើបទដ្ឋានអត្ថបទ
និងទិន្នន័យហ្វឹកហាត់ខុសៗគ្នា។ និក្ខេបបទនេះបង្ហាញការសិក្សាដែលមានការគ្រប់គ្រង លើការបណ្តុះបណ្តាលបន្ថែម
(fine-tuning) ម៉ូដែលសំឡេងដែលបានបណ្តុះបណ្តាលជាមុន សម្រាប់ភាសាខ្មែរ។ ការត្រួតពិនិត្យទាំងស្រុងលើសំណុំទិន្នន័យ
សំឡេងខ្មែរបើកចំហធំជាងគេ (៤៥០ ៣៩៦ ឃ្លា ១ ០៧៣,៩ ម៉ោង អ្នកនិយាយ ២០ នាក់) បង្ហាញថា ប្រយោគនីមួយៗ
ត្រូវបានអានដោយអ្នកនិយាយបួននាក់ ដូច្នេះការបែងចែកទិន្នន័យដោយចៃដន្យ នាំឲ្យប្រយោគតេស្តលេចចូលក្នុង
ទិន្នន័យហ្វឹកហាត់។ យើងបានបង្កើតការបែងចែកទិន្នន័យ ដែលដកចេញទាំងអ្នកនិយាយសម្រាប់តេស្ត និងប្រយោគទាំងអស់
ដែលពួកគេបានអាន ព្រមទាំងពិធីការធ្វើបទដ្ឋានអត្ថបទខ្មែរ និងអត្រាកំហុសតួអក្សរ (CER) ដែលមិនគិតដកឃ្លា។

ដោយគ្មានការបណ្តុះបណ្តាលបន្ថែម ម៉ូដែល Whisper បរាជ័យលើភាសាខ្មែរ (CER លើស ១០០%) ខណៈដែល MMS-1B
ជាមួយ adapter ខ្មែររបស់វា សម្រេចបាន ៩,៧១%។ ការបណ្តុះបណ្តាលបន្ថែមលើទិន្នន័យ ១០ ៥០ និង ១០០ ម៉ោង
ធ្វើឲ្យម៉ូដែលទាំងពីរប្រសើរឡើង៖ Whisper small ធ្លាក់ពី ១៩,៤៣% មក ៤,៧០% និងដល់ ២,៨៥% ជាមួយទិន្នន័យពេញ
៨៧២ ម៉ោង ហើយ MMS ធ្លាក់ពី ៨,៤៣% មក ៥,២៤%។ ការចាប់ផ្តើម adapter របស់ MMS ពី adapter ខ្មែរដែលមានស្រាប់
ជំនួសឲ្យការចាប់ផ្តើមពីសូន្យ កាត់បន្ថយ CER គ្រប់ទំហំទិន្នន័យ (មក ៤,៦២% នៅ ១០០ ម៉ោង) ហើយផ្តល់ម៉ូដែល
ដែលរឹងមាំបំផុត ដែលជាម៉ូដែលតែមួយគត់ ដែលល្អជាងម៉ូដែលដើមលើសំណុំតេស្ត FLEURS (១៤,៣៦% ធៀបនឹង ១៥,៤៩%)។
Whisper ឈ្នះលើសំឡេងប្រភេទដូចទិន្នន័យហ្វឹកហាត់ ប៉ុន្តែចាញ់លើសំឡេងប្រភេទផ្សេង (១៨,៦១% លើ FLEURS)
ហើយវាត្រឹមត្រូវជាងលើអ្នកនិយាយដែលវាធ្លាប់ឮក្នុងពេលហ្វឹកហាត់ ២៧–៣៤% ធៀបនឹង ៨–១១% សម្រាប់ MMS
ដូច្នេះការរក្សាអ្នកនិយាយសម្រាប់តេស្តឲ្យដាច់ពីទិន្នន័យហ្វឹកហាត់ គឺចាំបាច់សម្រាប់ការវាយតម្លៃដោយយុត្តិធម៌។
ការវិភាគកំហុសបង្ហាញថា កំហុសញឹកញាប់បំផុតរបស់ម៉ូដែលទាំងអស់ គឺការសរសេរអក្ខរាវិរុទ្ធពីររបៀប ដែលទិន្នន័យ
ខ្លួនឯងប្រើលាយឡំគ្នា ហើយលេខ ពាក្យអង់គ្លេស និងប្រយោគវែងៗ ជាចំណុចខ្សោយបំផុត។ ការបណ្តុះបណ្តាលទាំងអស់
ចំណាយប្រហែល ២៣ ដុល្លារអាមេរិក សម្រាប់ GPU ហើយកូដ ការកំណត់ ការបែងចែកទិន្នន័យ និងកំណត់ត្រាទាំងអស់
ត្រូវបានចែករំលែកជាសាធារណៈ។

**ពាក្យគន្លឹះ៖** ភាសាខ្មែរ ការសម្គាល់សំឡេងនិយាយដោយស្វ័យប្រវត្តិ ការបណ្តុះបណ្តាលបន្ថែម Whisper MMS
ភាសាដែលខ្វះធនធាន ពិធីការវាយតម្លៃ អត្រាកំហុសតួអក្សរ
