import os
import subprocess
import shutil
import sys
from typing import Sequence
from typing import Union

import numpy


if shutil.which('ffmpeg') is None:
    raise RuntimeError('command not found: ffmpeg. Please install')


def get_format(in_type: numpy.dtype):
    format_strings = [
        (numpy.float64, 'f64le'),
        (numpy.float32, 'f32le'),
        (numpy.int16, 's16le'),
        (numpy.int32, 's32le'),
        (numpy.uint32, 'u32le')]
    for dtype, string in format_strings:
        if in_type == dtype:
            return string
    raise RuntimeError(f'Not supported type: {in_type}')


def audio_ffmpeg(
        array: numpy.ndarray,
        nchannel: int=1,
        sampling_rate: int=16000,
        before_input_options: Sequence[str]=None,
        after_input_options: Sequence[str]=None,
        out_type: Union[numpy.dtype, str]=None,
        verbose: bool=False,
        timeout: float=None,
        ) -> numpy.ndarray:
    """
    Args:
        array (numpy.ndarray):
        nchannel (int): Input nchannel
        sampling_rate (int): Input sampling_rate
        after_input_options (Union[numpy.dtype, str]):
        before_input_options (Union[numpy.dtype, str]):
        out_type (numpy.dtype): Output dtype
        verbose (bool): Print command output
        timeout (float): If timeout is not None,
            after timeout[s] later, kill porcess
    """
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('command not found: ffmpeg. Please install')

    if not isinstance(array, numpy.ndarray):
        raise TypeError('Argument "array" must be numpy.ndarray')
    if not isinstance(nchannel, int):
        raise TypeError('Argument "nchannel" must be int')
    if not isinstance(sampling_rate, int):
        raise TypeError('Argument "sampling_rate" must be int')
    if after_input_options is None:
        after_input_options = []
    if any(not isinstance(opt, str) for opt in after_input_options):
        raise TypeError('Argument "options" must be a sequence of str')
    if before_input_options is None:
        before_input_options = []
    if any(not isinstance(opt, str) for opt in before_input_options):
        raise TypeError('Argument "options" must be a sequence of str')
    if not isinstance(verbose, bool):
        raise TypeError('Argument "verbose" must be bool')
    if out_type is None:
        out_type = array.dtype
    if isinstance(out_type, str):
        out_type = numpy.dtype(str)
    elif not isinstance(out_type, numpy.dtype):
        raise TypeError('Argument "out_type" must be numpy.dtype or str')

    format_string = get_format(array.dtype)
    command = ['ffmpeg',
               '-f', format_string, '-ar', str(sampling_rate), '-ac', str(nchannel)] +\
              list(before_input_options) + ['-i', 'pipe:0'] + list(after_input_options) +\
              ['-f', format_string, '-ar', str(sampling_rate), '-ac', str(nchannel), '-']
    with subprocess.Popen(
            command,
            bufsize=-1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE) as p:
        if verbose:
            print(' '.join(command), file=sys.stderr)

        try:
            stdout_bytes, stderr_bytes = p.communicate(
                input=array.tobytes(), timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout_bytes, stderr_bytes = p.communicate()
            if verbose:
                mes =\
                    f'TimeoutExpired: {timeout}[s]. ' \
                    f' '.join(command)
            else:
                mes =\
                    f'TimeoutExpired: {timeout}[s]. ' \
                    f'{" ".join(command) }{os.linesep}' \
                    f'{stderr_bytes.decode("utf-8")}'
            raise RuntimeError(mes)

        if p.returncode != 0 or stdout_bytes is None:
            if verbose:
                mes = ' '.join(command)
            else:
                mes = f"{' '.join(command) }{os.linesep}{stderr_bytes.decode('utf-8')}"
            raise RuntimeError(mes)

    audio = numpy.fromstring(stdout_bytes, dtype=array.dtype).astype(out_type)

    if nchannel > 1:
        audio = audio.reshape((-1, nchannel)).transpose()

    if audio.size == 0:
        return audio

    if out_type.kind == 'f':
        if normalize:
            peak = numpy.abs(audio).max()
        if peak > 0:
            audio /= peak
        elif in_type.kind == 'i':
            audio /= numpy.iinfo(array.dtype).max

    return audio


def audio_atempo(
        array: numpy.ndarray,
        tempo: float=1.0,
        nchannel: int=1,
        sampling_rate: int=16000,
        out_type: Union[numpy.dtype, str]=None,
        verbose: bool=False,
        timeout: float=None,
        ) -> numpy.ndarray:
    """
    Args:
        array (numpy.ndarray):
        tempo (float):
        nchannel (int): Input nchannel
        sampling_rate (int): Input sampling_rate
        out_type (numpy.dtype): Output dtype
        verbose (bool): Print command output
        timeout (float): If timeout is not None,
            after timeout[s] later, kill porcess
    """
    if tempo == 1.0:
        options = []
    else:
        options = ['-af', f'atempo={tempo}']
    return audio_ffmpeg(
        array=array,
        nchannel=nchannel,
        sampling_rate=sampling_rate,
        after_input_options=options,
        out_type=out_type,
        verbose=verbose,
        timeout=timeout)


def audio_trim(
        array: numpy.ndarray,
        time_offset: float,
        duration: float,
        nchannel: int=1,
        sampling_rate: int=16000,
        out_type: Union[numpy.dtype, str]=None,
        verbose: bool=False,
        timeout: float=None,
        ) -> numpy.ndarray:
    """
    Args:
        array (numpy.ndarray):
        time_offset (float): set the start time offset [s]
        duration (float): record or transcode "duration" seconds of audio/video
        nchannel (int): Input nchannel
        sampling_rate (int): Input sampling_rate
        out_type (numpy.dtype): Output dtype
        verbose (bool): Print command output
        timeout (float): If timeout is not None,
            after timeout[s] later, kill porcess
    """
    return audio_ffmpeg(
        array=array,
        nchannel=nchannel,
        sampling_rate=sampling_rate,
        after_input_options=['-ss', str(time_offset), '-t', str(duration)],
        out_type=out_type,
        verbose=verbose,
        timeout=timeout)



if __name__ == '__main__':
    array = numpy.random.randn(1000).astype(numpy.int16)
    print(len(audio_atempo(array, 2.0, sampling_rate=16000, nchannel=1)))
    print(len(audio_trim(array, 0, 0.01, sampling_rate=16000, nchannel=1)))
