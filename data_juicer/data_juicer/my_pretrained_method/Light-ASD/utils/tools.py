import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, 'model')
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    args.trialPathAVA     = os.path.join(args.dataPathAVA, 'csv')
    args.audioOrigPathAVA = os.path.join(args.dataPathAVA, 'orig_audios')
    args.visualOrigPathAVA= os.path.join(args.dataPathAVA, 'orig_videos')
    args.audioPathAVA     = os.path.join(args.dataPathAVA, 'clips_audios')
    args.visualPathAVA    = os.path.join(args.dataPathAVA, 'clips_videos')
    args.trainTrialAVA    = os.path.join(args.trialPathAVA, 'train_loader.csv')

    if args.evalDataType == 'val':
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'val_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'val_orig.csv')  
        args.evalCsvSave  = os.path.join(args.savePath,     'val_res.csv') 
    else:
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'test_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'test_orig.csv')    
        args.evalCsvSave  = os.path.join(args.savePath,     'test_res.csv')
    
    os.makedirs(args.modelSavePath, exist_ok = True)
    os.makedirs(args.dataPathAVA, exist_ok = True)
    return args


def preprocess_AVA(args):
    # This preprocesstion is modified based on this [repository](https://github.com/fuankarion/active-speakers-context).
    # The required space is 302 G. 
    # If you do not have enough space, you can delate `orig_videos`(167G) when you get `clips_videos(85G)`.
    #                             also you can delate `orig_audios`(44G) when you get `clips_audios`(6.4G).
    # So the final space is less than 100G.
    # The AVA dataset will be saved in 'AVApath' folder like the following format:
    # ```
    # ├── clips_audios  (The audio clips cut from the original movies)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── clips_videos (The face clips cut from the original movies, be save in the image format, frame-by-frame)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── csv
    # │   ├── test_file_list.txt (name of the test videos)
    # │   ├── test_loader.csv (The csv file we generated to load data for testing)
    # │   ├── test_orig.csv (The combination of the given test csv files)
    # │   ├── train_loader.csv (The csv file we generated to load data for training)
    # │   ├── train_orig.csv (The combination of the given training csv files)
    # │   ├── trainval_file_list.txt (name of the train/val videos)
    # │   ├── val_loader.csv (The csv file we generated to load data for validation)
    # │   └── val_orig.csv (The combination of the given validation csv files)
    # ├── orig_audios (The original audios from the movies)
    # │   ├── test
    # │   └── trainval
    # └── orig_videos (The original movies)
    #     ├── test
    #     └── trainval
    # ```

    download_csv(args) # Take 1 minute 
    download_videos(args) # Take 6 hours
    extract_audio(args) # Take 1 hour
    extract_audio_clips(args) # Take 3 minutes
    extract_video_clips(args) # Take about 2 days

def download_csv(args): 
    # Take 1 minute to download the required csv files
    Link = "1C1cGxPHaJAl1NQ2i7IhRgWmdvsPhBCUy"
    cmd = "gdown --id %s -O %s"%(Link, args.dataPathAVA + '/csv.tar.gz')
    subprocess.call(cmd, shell=True, stdout=None)
    cmd = "tar -xzvf %s -C %s"%(args.dataPathAVA + '/csv.tar.gz', args.dataPathAVA)
    subprocess.call(cmd, shell=True, stdout=None)
    os.remove(args.dataPathAVA + '/csv.tar.gz')

def download_videos(args): 
    # Take 6 hours to download the original movies, follow this repository: https://github.com/cvdfoundation/ava-dataset
    for dataType in ['trainval', 'test']:
        fileList = open('%s/%s_file_list.txt'%(args.trialPathAVA, dataType)).read().splitlines()   
        outFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
        for fileName in fileList:
            cmd = "wget -P %s https://s3.amazonaws.com/ava-dataset/%s/%s"%(outFolder, dataType, fileName)
            subprocess.call(cmd, shell=True, stdout=None)

def extract_audio(args):
    # Take 1 hour to extract the audio from movies
    for dataType in ['trainval', 'test']:
        inpFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
        outFolder = '%s/%s'%(args.audioOrigPathAVA, dataType)
        os.makedirs(outFolder, exist_ok = True)
        videos = glob.glob("%s/*"%(inpFolder))
        for videoPath in tqdm.tqdm(videos):
            audioPath = '%s/%s'%(outFolder, videoPath.split('/')[-1].split('.')[0] + '.wav')
            cmd = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 %s -loglevel panic" % (videoPath, audioPath))
            subprocess.call(cmd, shell=True, stdout=None)


def extract_audio_clips(args):
    # Take 3 minutes to extract the audio clips
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    for dataType in ['train', 'val', 'test']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv'%(dataType)), engine='python')
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        audioFeatures = {}
        outDir = os.path.join(args.audioPathAVA, dataType)
        audioDir = os.path.join(args.audioOrigPathAVA, dic[dataType])
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)
        for entity in tqdm.tqdm(entityList, total = len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            start = insData.iloc[0]['frame_timestamp']
            end = insData.iloc[-1]['frame_timestamp']
            entityID = insData.iloc[0]['entity_id']
            insPath = os.path.join(outDir, videoKey, entityID+'.wav')
            if videoKey not in audioFeatures.keys():                
                audioFile = os.path.join(audioDir, videoKey+'.wav')
                sr, audio = wavfile.read(audioFile)
                audioFeatures[videoKey] = audio
            audioStart = int(float(start)*sr)
            audioEnd = int(float(end)*sr)
            audioData = audioFeatures[videoKey][audioStart:audioEnd]
            wavfile.write(insPath, sr, audioData)

def extract_video_clips(args):
    # Take about 2 days to crop the face clips.
    # You can optimize this code to save time, while this process is one-time.
    # If you do not need the data for the test set, you can only deal with the train and val part. That will take 1 day.
    # This procession may have many warning info, you can just ignore it.
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    for dataType in ['train', 'val', 'test']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv'%(dataType)))
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        outDir = os.path.join(args.visualPathAVA, dataType)
        audioDir = os.path.join(args.visualOrigPathAVA, dic[dataType])
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)
        for entity in tqdm.tqdm(entityList, total = len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            entityID = insData.iloc[0]['entity_id']
            videoDir = os.path.join(args.visualOrigPathAVA, dic[dataType])
            videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(videoKey)))[0]
            V = cv2.VideoCapture(videoFile)
            insDir = os.path.join(os.path.join(outDir, videoKey, entityID))
            if not os.path.isdir(insDir):
                os.makedirs(insDir)
            j = 0
            for _, row in insData.iterrows():
                imageFilename = os.path.join(insDir, str("%.2f"%row['frame_timestamp'])+'.jpg')
                V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
                _, frame = V.read()
                h = numpy.size(frame, 0)
                w = numpy.size(frame, 1)
                x1 = int(row['entity_box_x1'] * w)
                y1 = int(row['entity_box_y1'] * h)
                x2 = int(row['entity_box_x2'] * w)
                y2 = int(row['entity_box_y2'] * h)
                face = frame[y1:y2, x1:x2, :]
                j = j+1
                cv2.imwrite(imageFilename, face)