import wandb
import fire
import os
import regex as re
from loguru import logger

PROJECT = 'Data-Juicer Reports'
IDS = {
    'text': 'lxjhedh2',
    'image': 'm4ree4t1',
    'video': '5lsusuby',
    'audio': '4ijnw0n2',
}
DIFF_TH = 0.1

def main():
    for modality in IDS:
        logger.info(f'--------------- {modality} ---------------')
        work_dir = f'outputs/performance_benchmark_{modality}/'

        wandb.login()
        run = wandb.init(project=PROJECT,
                         id=IDS[modality],
                         resume='must')

        # collect results from logs
        log_pt = r'export_(.*?)_time_(\d*?).txt'
        log_dir = os.path.join(work_dir, 'log')
        log_files = os.listdir(log_dir)
        log_file = None
        for fn in log_files:
            if re.match(log_pt, fn):
                log_file = fn
                break
        if log_file is None:
            logger.warning('No log files found.')
            exit()
        log_file = os.path.join(log_dir, log_file)
        with open(log_file) as fin:
            log_content = fin.read()
        op_pt = r'OP \[(.*?)\] Done in (.*?)s'
        total_pt = r'All OPs are done in (.*?)s'
        op_data = re.findall(op_pt, log_content)
        total_data = re.findall(total_pt, log_content)

        res = dict(op_data)
        res['total'] = total_data[0]
        res = {key: float(res[key]) for key in res}

        # upload results and finish the run
        run.log(res)
        run.finish()

        # compare with the last run
        api = wandb.Api()
        api_run = api.run(f'{PROJECT}/{IDS[modality]}')
        run_history = api_run.history()
        last_record = run_history.iloc[-1]

        for op_name, time in op_data:
            last_time = last_record[op_name]
            dif = (time - last_time) / last_time
            if dif > 0.1:
                logger.warning(f'Time cost for OP {[op_name]} increased by '
                               f'{dif * 100}% (> 10%). Before-{last_time} vs. '
                               f'Now-{time}')
            else:
                logger.info(f'Time cost for OP {[op_name]} increased by '
                            f'{dif * 100}%. Before-{last_time} vs. Now-{time}')
        last_total = last_record['total']
        dif_total = (res['total'] - last_total) / last_total
        if dif_total > 0.1:
            logger.warning(f'Total time cost increased by {dif_total * 100}% '
                           f'(> 10%). Before-{last_total} vs. '
                           f'Now-{res["total"]}')
        else:
            logger.info(f'Total time cost increased by {dif_total * 100}%. '
                        f'Before-{last_total} vs. Now-{res["total"]}')


if __name__ == '__main__':
    fire.Fire(main)
