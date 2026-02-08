import json
import glob
import argparse
import os
import re
from collections import defaultdict

def get_base_route_id(route_id):
    return re.sub(r'_rep\d+$', '', route_id)

def merge_route_json(folder_path):
    file_paths = glob.glob(f'{folder_path}/*.json')
    merged_records = []
    driving_score = []
    success_num = 0
    route_stats = defaultdict(lambda: {"count": 0, "success": 0, "score_sum": 0.0})

    for file_path in file_paths:
        if 'merged.json' in file_path: continue
        with open(file_path) as file:
            data = json.load(file)
            records = data['_checkpoint']['records']
            for rd in records:
                rd.pop('index')
                score = rd['scores']['score_composed']
                merged_records.append(rd)
                route_id = rd['route_id']
                base_route_id = get_base_route_id(route_id)
                route_stats[base_route_id]["count"] += 1
                route_stats[base_route_id]["score_sum"] += score
                
                driving_score.append(rd['scores']['score_composed'])
                is_success = False
                if rd['status']=='Completed' or rd['status']=='Perfect':
                    success_flag = True
                    for k,v in rd['infractions'].items():
                        if len(v)>0 and k != 'min_speed_infractions':
                            success_flag = False
                            break
                    if success_flag:
                        is_success = True
                        success_num += 1
                        # print(rd['route_id'])
                if is_success:
                    route_stats[base_route_id]["success"] += 1
    import csv
    sorted_routes = sorted(route_stats.items(), key=lambda x: x[0])
    with open('route_stats.tsv', 'w', encoding='utf-8') as tsvfile:
        tsvfile.write("route_id\tsuccess_rate\tavg_score\tcount\n")
        for base_route_id, stat in sorted_routes:
            count = stat["count"]
            success = stat["success"]
            score_avg = stat["score_sum"] / count if count > 0 else 0
            success_rate = success / count if count > 0 else 0
            tsvfile.write(f"{base_route_id}\t{success_rate:.2f}\t{score_avg:.2f}\t{count}\n")
        merged_records = sorted(merged_records, key=lambda d: d['route_id'], reverse=True)
    print(len(route_stats))
    print("\nPer-base-route stats:")
    for base_route_id, stat in route_stats.items():
        count = stat["count"]
        success = stat["success"]
        score_avg = stat["score_sum"] / count if count > 0 else 0
        success_rate = success / count if count > 0 else 0
        print(f"route_id: {base_route_id} | success_rate: {success_rate:.2f} | avg_score: {score_avg:.2f} | count: {count:.1f}")
    
    _checkpoint = {
        "records": merged_records
    }
    print("driving score:", sum(driving_score) / len(merged_records))
    print("success rate:", success_num / len(merged_records))
    merged_data = {
        "_checkpoint": _checkpoint,
        "driving score": sum(driving_score) / len(merged_records),
        "success rate": success_num / len(merged_records),
        "eval num": len(driving_score),
    }

    with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
        json.dump(merged_data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='old foo help')
    args = parser.parse_args()
    merge_route_json(args.folder)

