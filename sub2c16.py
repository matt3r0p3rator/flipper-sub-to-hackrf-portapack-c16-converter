import os
import argparse
import math
import numpy as np
import psutil
import signal
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, Future

SUPPORTED_PROTOCOLS = ['RAW']
HACKRF_OFFSET = 0
TARGET_CPU_USAGE = 50  # Target CPU usage in percentage
CPU_CHECK_INTERVAL = 2  # Seconds between CPU usage checks

# Graceful shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully by setting a shutdown flag."""
    global shutdown_requested
    print("\nShutdown requested. Killing all tasks...")
    shutdown_requested = True


# Attach the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


# Function to parse Flipper Zero .sub files
def parse_Sub(file: str) -> dict:
    try:
        with open(file, 'r') as f:
            sub_data = f.read()
    except Exception as e:
        print(f'Error reading input file: {e}')
        return None

    sub_chunks = [r.strip() for r in sub_data.split('\n') if r.strip()]
    try:
        info = {k.lower(): v.strip() for k, v in (row.split(':') for row in sub_chunks[:5])}
    except ValueError:
        print(f"Failed to parse metadata from {file}. Ensure the .sub file has the correct format.")
        return None

    try:
        info['chunks'] = [
            list(map(int, r.split(':')[1].split()))
            for r in sub_chunks[5:]
            if ':' in r
        ]
    except Exception as e:
        print(f"Failed to parse data chunks in {file}: {e}")
        return None

    return info


# Save non-RAW data files as TXT with useful information
def save_non_raw_file(input_file: str, output_dir: str, info: dict, verbose: bool):
    output_file = os.path.join(output_dir, f"!{os.path.basename(input_file)}.TXT")
    try:
        with open(output_file, 'w') as f:
            f.write(f"# Metadata for {input_file}\n")
            for key, value in info.items():
                if key != 'chunks':  # Write metadata first
                    f.write(f"{key.capitalize()}: {value}\n")
            f.write("\n# Data Chunks\n")
            for chunk in info.get('chunks', []):  # Append data chunks
                f.write(f"{chunk}\n")

        if verbose:
            print(f"Saved non-RAW file metadata to {output_file}")
    except Exception as e:
        print(f"Error saving non-RAW file {input_file}: {e}")


# Convert durations to binary IQ sequence (chunk-based)
def durations_to_bin_sequence(durations: List[List[int]], sampling_rate: int, intermediate_freq: int, amplitude: int) -> List[Tuple[int, int]]:
    sequence = []
    for chunk in durations:
        for duration in chunk:
            sequence.extend(us_to_sin(duration > 0, abs(duration), sampling_rate, intermediate_freq, amplitude))
            if len(sequence) > 100000:  # Process in manageable chunks
                yield sequence
                sequence = []
    if sequence:
        yield sequence


# Generate sine wave from duration and sampling parameters
def us_to_sin(level: bool, duration: int, sampling_rate: int, intermediate_freq: int, amplitude: int) -> List[Tuple[int, int]]:
    iterations = int(sampling_rate * duration / 1_000_000)
    if iterations == 0:
        return []

    data_step_per_sample = 2 * math.pi * intermediate_freq / sampling_rate
    hackrf_amplitude = (256 ** 2 - 1) * (amplitude / 100)

    return [
        (
            HACKRF_OFFSET + int(math.floor(math.cos(i * data_step_per_sample) * (hackrf_amplitude / 2))),
            HACKRF_OFFSET + int(math.floor(math.sin(i * data_step_per_sample) * (hackrf_amplitude / 2)))
        )
        if level else (HACKRF_OFFSET, HACKRF_OFFSET)
        for i in range(iterations)
    ]


# Convert IQ sequence to 16LE binary buffer (chunk-based)
def sequence_to_16LEBuffer(sequence: List[Tuple[int, int]]) -> bytes:
    return np.array(sequence).astype(np.int16).tobytes()


# Function to write HackRF .C16 and metadata .TXT files
def write_HRF_file(file: str, buffer_generator, frequency: str, sampling_rate: int):
    c16_path = f'{file}.C16'
    txt_path = f'{file}.TXT'

    with open(c16_path, 'wb') as f:
        for buffer_chunk in buffer_generator:
            f.write(sequence_to_16LEBuffer(buffer_chunk))

    with open(txt_path, 'w') as f:
        f.write(generate_meta_string(frequency, sampling_rate))

    return c16_path, txt_path


# Metadata generator
def generate_meta_string(frequency: str, sampling_rate: int) -> str:
    meta = [['sample_rate', sampling_rate], ['center_frequency', frequency]]
    return '\n'.join('='.join(map(str, r)) for r in meta)


# Process a single .sub file
def process_file(input_file: str, output_dir: str, sampling_rate: int, intermediate_freq: int, amplitude: int, verbose: bool):
    if shutdown_requested:
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the output directory if it doesn't exist

    info = parse_Sub(input_file)
    if not info:  # If the file fails to parse
        try:
            output_file = os.path.join(output_dir, f"!{os.path.basename(input_file)}")
            with open(input_file, 'rb') as src, open(output_file, 'wb') as dst:
                dst.write(src.read())
            if verbose:
                print(f"Failed to parse {input_file}. Copied to {output_file}.")
        except Exception as e:
            print(f"Error copying failed file {input_file}: {e}")
        return

    output_base = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0])
    if info.get('protocol') not in SUPPORTED_PROTOCOLS:
        save_non_raw_file(input_file, output_dir, info, verbose)
    else:
        if verbose:
            print(f"Processing {input_file} -> {output_dir}")

        chunks = info['chunks']
        buffer_generator = durations_to_bin_sequence(chunks, sampling_rate, intermediate_freq, amplitude)
        try:
            c16_path, txt_path = write_HRF_file(output_base, buffer_generator, info.get('frequency', '433920000'), sampling_rate)
            if verbose:
                print(f"Written files: {c16_path}, {txt_path}")
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")


# Process all .sub files in a folder
def process_folder(input_folder: str, output_folder: str, sampling_rate: int, intermediate_freq: int, amplitude: int, verbose: bool, max_threads: int):
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if file.endswith('.sub')
    ]

    with ThreadPoolExecutor(max_threads) as executor:
        tasks = [
            executor.submit(
                process_file,
                file,
                os.path.join(output_folder, os.path.relpath(os.path.dirname(file), input_folder)),
                sampling_rate,
                intermediate_freq,
                amplitude,
                verbose
            )
            for file in all_files
        ]
        try:
            # Monitoring the tasks
            for task in as_completed(tasks, timeout=5):
                if shutdown_requested:
                    # Cancel all tasks immediately when shutdown is requested
                    print("\nKilling all tasks...")
                    for future in tasks:
                        future.cancel()  # Force kill tasks immediately
                    break
        except TimeoutError:
            print("\nTimeout reached. Exiting after ongoing tasks.")


# Parse script arguments
def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Convert Flipper Zero .sub files to HackRF .C16 files")
    parser.add_argument('input_folder', help="Input folder containing .sub files")
    parser.add_argument('output_folder', help="Output folder for converted files")
    parser.add_argument('-sr', '--sampling_rate', type=int, default=500000, help="Sampling rate for output. Default: 500ks/s")
    parser.add_argument('-if', '--intermediate_freq', type=int, default=None, help="Intermediate frequency. Default: sampling_rate / 100")
    parser.add_argument('-a', '--amplitude', type=int, default=100, help="Amplitude percentage. Default: 100%")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-t', '--threads', type=int, default=8, help="Maximum number of threads. Default: 8")
    return vars(parser.parse_args())


# Main script execution
if __name__ == "__main__":
    args = parse_args()

    input_folder = args['input_folder']
    output_folder = args['output_folder']
    sampling_rate = args['sampling_rate']
    intermediate_freq = args['intermediate_freq'] or sampling_rate // 100
    amplitude = args['amplitude']
    verbose = args['verbose']
    max_threads = args['threads']

    process_folder(input_folder, output_folder, sampling_rate, intermediate_freq, amplitude, verbose, max_threads)
