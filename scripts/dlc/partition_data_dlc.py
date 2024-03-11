import argparse
import json
import os
from collections import defaultdict
from typing import List


def partition_data(json_file_path: str, hostnames: List[str]):
    with open(json_file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    video_to_entries_map = defaultdict(list)
    for entry in data:
        video_path = entry['videos'][0]
        video_to_entries_map[video_path].append(entry)
    nodes_data = defaultdict(list)
    nodes_video_size = {k: 0 for k in hostnames}

    # distribute videos to nodes based on the total size of videos
    video_sizes = {
        video: os.path.getsize(video)
        for video in video_to_entries_map.keys()
    }

    sorted_videos = sorted(video_sizes, key=video_sizes.get, reverse=True)
    for video in sorted_videos:
        min_node = min(nodes_video_size, key=nodes_video_size.get)
        nodes_data[min_node].extend(video_to_entries_map[video])
        nodes_video_size[min_node] += video_sizes[video]

    for hostname in hostnames:
        host_file_path = f"{json_file_path.rsplit('.', 1)[0]}_{hostname}.jsonl"
        with open(host_file_path, 'w') as f:
            for entry in nodes_data[hostname]:
                f.write(json.dumps(entry) + '\n')


# Usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Partition data across hostnames.')

    parser.add_argument('file_path',
                        type=str,
                        help='Path of the file to distribute.')
    parser.add_argument('hostnames', nargs='+', help='The list of hostnames')

    args = parser.parse_args()

    partition_data(args.file_path, args.hostnames)
