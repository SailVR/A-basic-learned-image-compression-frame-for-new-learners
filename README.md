# A basic learned image compression frame for new learners

this is a basic learned image compression frame

Base on [CompressAI](https://github.com/InterDigitalInc/CompressAI/) and [liujiahneg&#39;s frame](https://github.com/liujiaheng/compression)

including:

encoder，decoder，context model，hyper，and so on

# dataset

train dataset：you can use fiftyone to download open-images，this is downloader_openimages.py from [STF](https://github.com/googolxx/stf)

```
import fiftyone

if __name__ == '__main__':
    """Download the training/test data set from OpenImages."""

    dataset_train = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="train",
        max_samples=300000,
        label_types=["classifications"],
        dataset_dir='openimages',
    )
    dataset_test = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="test",
        max_samples=10000,
        label_types=["classifications"],
        dataset_dir='openimages',
    )
```

test_dateset: [kodak](http://r0k.us/graphics/kodak/)

# Train

for mse：

```
python train.py  --metrics mse --lmbda 0.013 --train /path/train/ --val /path/val/ -p /path/pretrained_model/
```

for ms-ssim：

```
python train.py  --metrics ms-ssim --lmbda 8.73 /path/train/ --val /path/val/ -p /path/pretrained_model/
```

The train lmbda are follows：

| Quality | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      |
| ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| MSE     | 0.0018 | 0.0035 | 0.0067 | 0.0130 | 0.0250 | 0.0483 | 0.0932 | 0.1800 |
| MS-SSIM | 2.40   | 4.58   | 8.73   | 16.64  | 31.73  | 60.50  | 115.37 | 220.00 |

# Test

```
python test.py /path/train/ --val /path/val/ -p /path/pretrained_model/
```
pretrained model can be download in [this](https://pan.baidu.com/s/1EOcOmBd-dOHiQLEc4AA0uQ ) code:jhnx
The result of test_img.py is same as test.py

# Other

when you change one py file in models/

you can run

```
python3 -m models.analysis

```

# Contact

sailruan@126.com
