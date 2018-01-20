# python_audio_ffmpeg
Python ffmpeg wrapper to convert from numpy.ndarray to numpy.ndarray

## Requirements

```
python3.6
numpy
ffmpeg
```

## Install
```bash
conda install ffmpeg
python setup.py install
```

## Usage

```python
import numpy
from audio_ffmpeg import ffmpeg

array = numpy.random.randn(1000).astype(numpy.int16)
audio_atempo(array, 2.0, sampling_rate=16000, nchannel=1)
audio_trim(array, 0, 0.01, sampling_rate=16000, nchannel=1)
```
