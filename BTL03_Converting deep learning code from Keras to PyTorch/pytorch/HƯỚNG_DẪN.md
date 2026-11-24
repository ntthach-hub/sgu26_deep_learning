# BTL03 · Chạy các ví dụ PyTorch

Mở PowerShell trong thư mục BTL03 (thư mục chứa `pytorch`).

```powershell
python -m pip install -r pytorch/requirements.txt
python pytorch/run.py environment
python pytorch/run.py --help
```

Chạy từng lệnh theo notebook cần đối chiếu.

| Notebook gốc | Lệnh |
|---|---|
| `0. Getting Started/getting_started.ipynb` | `python pytorch/run.py environment` |
| `1. MLP/Func_apxm_MLP.ipynb` | `python pytorch/run.py gaussian` |
| `1. MLP/MNIST_MLP.ipynb` | `python pytorch/run.py mnist_mlp` |
| `2. CNN + MLP/1_MNIST_CNN.ipynb` | `python pytorch/run.py mnist_cnn` |
| `2. CNN + MLP/2_Human_Gender_CNN.ipynb` | `python pytorch/run.py gender_cnn` |
| `2. CNN + MLP/3_Emotion_Reg_CNN.ipynb` | `python pytorch/run.py emotion_cnn` |
| `3. CNN + LSTM/MNIST_CNN_LSTM.ipynb` | `python pytorch/run.py digit_sum` |
| `4. Training Utilities/2_Human_Gender_CNN.ipynb` | `python pytorch/run.py gender_callbacks` |
| `4. Training Utilities/3_Emotion_Reg_CNN.ipynb` | `python pytorch/run.py emotion_augmentation` |
| `5. Others/1_segnet.ipynb` | `python pytorch/run.py segnet` |
| `5. Others/2_obj_detection.ipynb` | `python pytorch/run.py vehicle` |

`Iris_Bayesian_decision_making.ipynb` dùng scikit-learn và `colab_getting_started.ipynb` hướng dẫn Colab, đã có lệnh cài PyTorch. Hai notebook này dùng trực tiếp trong `examples`.

## Dữ liệu

Git chỉ theo dõi `datasets/.gitkeep`. Sau khi clone, lấy dữ liệu theo các nguồn bên dưới và đặt vào `datasets/`.

- MNIST dùng `datasets/mnist.npz` nếu có. Bộ dữ liệu của Yann LeCun và Corinna Cortes được phân phối dưới giấy phép CC BY-SA 3.0. PyTorch đọc bốn mảng trong bản MNIST do Keras phân phối. Khi thiếu tệp này, torchvision tự tải MNIST vào thư mục dữ liệu. [Nguồn MNIST và SHA-256](https://raw.githubusercontent.com/keras-team/keras/master/keras/src/datasets/mnist.py)
- Để chạy lại SegNet và phát hiện xe, lấy `depth_hand.zip`, `car_data.zip` và `test.jpg` từ `examples.zip` của đề bài. Đặt cả ba tệp vào `examples/5. Others/` trong thư mục BTL03 và giữ hai tệp dữ liệu ở dạng ZIP. SegNet dùng `depth_hand.zip`. Ví dụ xe dùng `car_data.zip` để huấn luyện và `test.jpg` để tạo heatmap. Các nguồn ghi trong notebook gốc gồm [ảnh độ sâu bàn tay](https://drive.google.com/file/d/1MRiBnDW9dE7mNb5bL-jzhmtTd2CSklvL/view?usp=sharing) và [ảnh xe](https://drive.google.com/file/d/1rqitUFrFhB84tFVtg1Ad1TVz6coCWpNZ/view?usp=sharing).
- Bốn ví dụ chân dung và FER2013 dùng các tệp dưới đây. Hai thư mục `datasets/wiki_quick` và `datasets/fer2013_quick` chứa mẫu được lấy từ dữ liệu gốc để tái lập các lượt chạy CPU trong báo cáo. Để chạy cả bộ, tải các tệp gốc và đặt trong `datasets` hoặc trỏ `--data` tới thư mục chứa chúng. [WIKI theo notebook](https://drive.google.com/drive/folders/0BxINLo5jshCRYW8xODhNSlkyLTQ?resourcekey=0-RNJvvSSE67Q1bMqQi3A1RQ&usp=sharing) · [FER2013 theo kho mã của tác giả](https://drive.google.com/drive/folders/15EozdYlGeO3d-ny8nbqFM3ojTGsj2hdE?usp=sharing)

```text
datasets/
  64_64_11938_4098_train_x_onehot.npy
  64_64_11938_4098_train_y_onehot.npy
  64_64_5968_4098_val_x_onehot.npy
  64_64_5968_4098_val_y_onehot.npy
  fer2013_training_onehot.csv
  fer2013_publictest_onehot.csv
```

Ảnh chân dung có kích thước 64 × 64 với một kênh. Mỗi tệp nhãn NPY có hai cột one-hot, theo thứ tự `female, male`. Mỗi dòng CSV FER2013 có 2.304 giá trị pixel và 7 cột one-hot, theo thứ tự `angry, disgust, fear, happy, sad, surprise, neutral`. Dùng CSV số phân cách bằng dấu phẩy, không tiêu đề, theo định dạng của notebook gốc.

Mẫu WIKI lấy 128 ảnh đầu mỗi lớp từ tập train và 64 ảnh đầu mỗi lớp từ tập validation. Mẫu FER2013 lấy 384 dòng đầu của CSV train và 192 dòng đầu của CSV public test. Tên tệp trong hai thư mục lấy mẫu giữ nguyên theo notebook để dùng với cùng bộ đọc dữ liệu.

## Chạy nhanh và đọc kết quả

```powershell
python pytorch/run.py mnist_cnn --epochs 2 --limit 512 --device cpu --out results_quick
python pytorch/run.py gender_callbacks --data datasets/wiki_quick --epochs 9 --batch-size 16 --threads 2
python pytorch/run.py emotion_augmentation --data datasets/fer2013_quick --epochs 3 --batch-size 16 --threads 3
tensorboard --logdir results/gender_callbacks/tensorboard
```

Các lệnh mặc định dùng cấu hình huấn luyện theo notebook gốc và tự chọn CUDA khi PyTorch phát hiện GPU. `--limit` lấy ngẫu nhiên tối đa số mẫu được chỉ định **trong mỗi tập** với seed cố định. Tập đánh giá đi theo notebook gốc. Khi tập này được dùng trong từng epoch hoặc để chọn checkpoint, nên gọi kết quả là chỉ số trên tập đánh giá.

Mỗi lần chạy ghi cấu hình JSON, lịch sử CSV, biểu đồ huấn luyện và hình dự đoán vào `results/<ví dụ>/` hoặc thư mục chỉ định bằng `--out`. Lệnh huấn luyện cũng tạo checkpoint `best.pt` theo validation loss và `last.pt` cho epoch cuối. Lệnh `gender_callbacks` nạp lại `best.pt` rồi đánh giá. Dùng `--out` khác cho mỗi thí nghiệm cần lưu riêng. Gaussian xuất `sweeps.csv`, hình bề mặt Gaussian của hai tập và đồ thị quét siêu tham số. Báo cáo Word ghi các lệnh đã chạy để tạo số liệu và hình trong `results`.

`models.py` chứa các kiến trúc. `data.py` xử lý dữ liệu và nhãn. `training.py` quản lý vòng lặp và callback. `gaussian.py` chạy ba lượt quét. `previews.py` lưu hình minh họa và PCA. Các notebook gốc để đối chiếu nằm trong `examples.zip` của đề bài.
