# Indian Language Classification

## Steps to Submit the Script on HTCondor

### 1. SSH into the Server

```bash
ssh neuronet_team239@conduit2.hpc.uni-saarland.de
```

### 2. Switch to nnti_project directory

.py and .sub files are already saved in `runlogs/` directory, so just need to switch to this directory:

```bash
cd nnti_project
```

### 3. Create the Output Log Directory

HTCondor writes logs to `runlogs/`, which must exist before submitting:

```bash
mkdir -p runlogs
```

### 4. Submit the Job

```bash
condor_submit audio_ml.sub
```

### 5. Monitor the Job

Check job status:

```bash
condor_q
```

Watch live log output:

```bash
tail -f runlogs/audio_ml.*.log
```

Check stdout/stderr:

```bash
tail -f runlogs/audio_ml.*.out
tail -f runlogs/audio_ml.*.err
```

### 7. Retrieve Output

After the job completes, HTCondor transfers `indic-SLID/inprogress/` back to submission directory. Copy it to your local machine:

```bash
scp -r neuronet_team239@conduit2.hpc.uni-saarland.de:~/nnti_project/inprogress .
```

The output directory will contain:
- Saved model
- `confusion_matrix.png`
- `tsne.png`