# Dataset Inventory & Feasibility Audit: Dog-Human Social Play Bioacoustic Dataset

**Project:** Zero Rabies-MMNet  
**Stage:** Video Preprocessing & Temporal Pipeline Design  
**Date:** 2026-09-23  
**Status:** Audit Completed / Not Yet Locally Downloaded  

---

## 1. Executive Summary & Local Disk Audit Findings

An exhaustive search across the project directories (`Zero-Rabies-MMNet/data/raw/`, `Dataset/`, `source_code_senior/`, and system user directories) was conducted to locate the **Dog-Human Social Play Bioacoustic Dataset** referenced in the project interim dataset guide.

### Key Finding on Local Availability:
1. **Local Disk Status:** The Dog-Human Social Play dataset files (`01 Samples of play sessions.zip`, `02 Raw recordings.zip`, `03 Play session recordings.zip`, `04 Annotations.zip`, `05 Metadata.csv`) **are NOT currently present** in `data/raw/` or anywhere on the local storage drives.
2. **Current `data/raw/` Contents:**
   - `animal_kingdom/`: 30,100 action recognition videos and annotations (68 model-eligible dog clips).
   - `dog_pose/`: 8,476 images with 24-keypoint YOLO Pose annotations.
3. **Other Local Workspace Media (`Dataset/` folder):**
   - Contains 6 unannotated YouTube video/audio download samples (`11.mp4`–`16.mp4`, `11.wav`–`16.wav`). These are demo test recordings, not the research dataset.
4. **Authoritative Dataset Origin Located:**
   - **Title:** *"Dog play pant: An annotated dataset of dog vocalizations during dog-human social play"*
   - **Permanent Zenodo Archive:** [DOI 10.5281/zenodo.18972388](https://doi.org/10.5281/zenodo.18972388)
   - **Research Preprint:** [bioRxiv DOI 10.64898/2026.04.20.719471](https://doi.org/10.64898/2026.04.20.719471) (April 2026)
   - **Originating Institution:** BARKS Lab, Department of Ethology, Eötvös Loránd University, Budapest, Hungary & SCAN Unit, University of Vienna.
   - **Companion Code Repository:** [GitHub: rhernandez00/bioacoustic-dataset](https://github.com/rhernandez00/bioacoustic-dataset) (contains feature extraction pipeline and `database_analysis.ipynb`).

---

## 2. Dataset Structure & Specifications (from Published Artifacts)

| Attribute | Specification |
|---|---|
| **Archive Identifier** | Zenodo Record `18972388` |
| **Total Play Sessions** | 30 sessions |
| **Unique Dogs / Participants** | 17 individual pet dogs (aged 6–24 months, diverse breeds) |
| **Sessions per Dog** | 1 to 3 sessions per dog |
| **Total Analyzed Audio Duration** | 7,482 seconds (~2.08 hours) |
| **Session Duration Range** | 34 s to 613 s (Mean: 249.40 s / ~4.16 min) |
| **Video Recording Setup** | Multi-view synchronized camera system (7 Basler a2A 1920-51gcPRO IP cameras) |
| **Video Format & Angles** | Synchronized multi-view video (2 primary angles of view + sound) |
| **Audio Recording Setup** | Zoom H4 Essential recorder with Rode Wireless Go 2 clip-on, Sennheiser ME66 shotgun mic, Sennheiser ME64 cardioid mic |
| **Audio Format** | WAV format, PCM encoding, 48 kHz sampling rate, 16-bit / 32-bit |
| **Audio Channels** | Predominantly Mono (1-channel); stereo (2-channel) for sessions P-05 and P-06 |
| **Metadata File** | `05 Metadata.csv` (semicolon-delimited): dog ID, session ID, date, name, breed, sex, age, weight, recording equipment, mic channels |

---

## 3. Ground-Truth Annotations & Labels

The dataset provides two structured annotation layers marking precise temporal boundaries (`Begin Time (s)`, `End Time (s)`):
- **Layer 1:** Initial annotations exported from Raven Lite 2.0.5 (`04 Annotations.zip`).
- **Layer 2:** Expert-reviewed annotations validated in Praat TextGrid format by bioacousticians.

### Actual Annotated Sound Categories (Acoustic Ethogram):

| Code | Sound Category | Description in Ethogram |
|---|---|---|
| **A** | `Growl` | Low-frequency aggressive/play vocalization |
| **B** | `Whine` | Tonal, high-pitched vocalization |
| **C** | `Bark/yelp` | Short, loud, explosive vocalization |
| **D** | `Moan` | Sustained low-to-mid frequency vocalization |
| **E** | `Pant` | Rhythmic inhalation/exhalation; play pant bouts |
| **F** | `Cough/woof` | Short burst sound |
| **G** | `Howl` | Sustained tonal acoustic vocalization |
| **H** | `Grunt` | Short guttural low-frequency sound |
| **I** | `Other` | Unclassified dog sound |
| **J** | `Sneeze` | Non-vocal communicative signal (play signal) |
| **K** | `Shake` | Non-vocal body shake (behavioral transition marker) |
| **L** | `Human vocalizations` | Caregiver speech/vocal encouragement |

*(Note: In the companion analysis `database_analysis.ipynb`, label `M` was excluded during coding for ambiguity).*

---

## 4. Usability Assessment: Can Social Play Train Mamba for Behavioural Proxy?

### Critical Evaluation:
1. **Mismatch with Video Action Labels:**
   - The student guide suggested: *"we treat 'high-energy rough play' as our stand-in for agitated behavior and 'calm interaction' as our stand-in for normal behavior."*
   - However, **inspection of the actual dataset reveals that ground-truth labels are BIOACOUSTIC SOUND SEGMENTS** (`Growl`, `Pant`, `Bark`, `Whine`, etc.), **NOT frame-by-frame visual action labels** (e.g. no "rough play" vs "calm interaction" bounding boxes or sequence tags).
   - The entire recorded session represents an active play session between a dog and familiar human in a 5.46 m × 4.41 m laboratory room.
2. **Video Role:**
   - In the official descriptor, video is provided for *contextual reference* and multi-view synchronization. It does not have pre-annotated bounding boxes, pose keypoints, or behavioral state transition boundaries.
3. **Audio Role (Block 4 & Block 5 Alignment):**
   - The dataset is exceptionally well-suited for **audio distress / play vocalization modeling** (Block 4) and **synchronized audio-visual temporal alignment** (Block 5).
   - Because video and audio are synchronized at 48 kHz / multi-view IP cameras, it serves as the primary benchmark for cross-modal temporal fusion.
4. **Immediate Video Temporal (Mamba) Training Feasibility:**
   - Social Play cannot immediately train the video Mamba on "calm vs agitated" motion without first creating or deriving video-level proxy segment labels from the audio event timestamps (e.g. associating barking/growling play bouts with agitation and silent/panting intervals with calmer states).
   - In contrast, **Animal Kingdom** already provides verified, audited frame-level canine action labels (`Running`, `Attacking`, `Jumping` $\rightarrow$ `AGITATED`; `Walking`, `Keeping still`, `Yawning` $\rightarrow$ `CALM`).

---

## 5. Session and Dog Structure & Leakage Prevention Policy

### Structure:
- **17 unique dogs** (`participant_id`: P-01 to P-17).
- **30 total sessions** (`session_id`: 1 to 3 per dog).
- Multiple recordings originate from the exact same dog and familiar human handler.

### Mandatory Split Policy:
- **STRICT GROUPED SPLITTING (Dog-Level Separation):**
  - Clips or frames from the same `participant_id` (dog) must NEVER appear in both training and test partitions.
  - In the companion analysis (`database_analysis.ipynb`), the original authors enforce `GroupKFold(groups=participant_id)` for this exact reason.
  - Random shuffling of clips or temporal windows would result in severe identity and acoustic environment leakage.
- **Animal Kingdom Independence:**
  - The 21 held-out Animal Kingdom dog test clips must remain completely untouched. Social Play data must not contaminate Animal Kingdom evaluation.

---

## 6. Actionable Next Steps for Dog-Human Social Play
1. **Download Requirement:** When authorized, download Zenodo record `18972388` (`01 Samples of play sessions.zip`, `04 Annotations.zip`, `05 Metadata.csv`, `03 Play session recordings.zip`) into `Zero-Rabies-MMNet/data/raw/social_play/`.
2. **Synchronized Benchmark:** Use the 17 10-second sample clips (`01 Samples`) as the initial verification set for synchronized audio-video feature extraction before processing large raw multi-view archives.
3. **Audio Tier Integration:** Map sound labels (`Growl`, `Bark`, `Pant`, `Whine`) into the Audio Classifier (Block 4) alongside DogSpeak and Barkopedia.
