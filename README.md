# How to run

1. **Install dependency dan environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pre-install dataset dan weight InceptionV3 untuk perhitungan FID** *(optional, otomatis install saat run main)*
   ```bash
   python scripts/download_assets.py
   ```

3. **Run main pipeline**
   ```bash
   python -u main.py
   ```
   Menggunakan `seed=42`, stratified 80% Train (56.000) / 10% Validation (7.000) / 10% Test (7.000), dan `batch_size=128`.

4. **Unit test untuk model**
   ```bash
   python -m pytest tests/test_models.py -v
   ```

# Notes

Projek dikerjakan secara fully lokal hingga selesai, dengan GPU NVIDIA GeForce RTX 4060 (8,188 MiB / ~8 GB VRAM)
