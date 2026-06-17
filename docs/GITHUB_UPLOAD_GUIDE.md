# How to put DIEP-ThermoSim on GitHub: step-by-step

This walks through every step from "nothing on GitHub yet" to "polished public repo pinned on your profile and referenced in your application." Time required: about 45 minutes the first time, 15 minutes for subsequent projects.

---

## Prerequisites (one-time setup — skip if already done)

### Install Git locally
On Windows: download from https://git-scm.com/download/win and accept defaults.
On macOS: `xcode-select --install` in Terminal.
On Ubuntu/Debian: `sudo apt install git`.

Verify: `git --version` should print a version number.

### Configure your Git identity
Open a terminal and run, with **your** name and the email you use for GitHub:

```bash
git config --global user.name "Iqra Asghar"
git config --global user.email "your_github_email@example.com"
```

This is the identity that will be attached to every commit. Use the same email as your GitHub account so commits are linked to your profile.

### Set up SSH authentication (recommended)
GitHub deprecated password authentication. The easiest reliable option is SSH.

```bash
# generate a key (press Enter at every prompt to accept defaults)
ssh-keygen -t ed25519 -C "your_github_email@example.com"

# copy the public key
# macOS:
pbcopy < ~/.ssh/id_ed25519.pub
# Linux:
cat ~/.ssh/id_ed25519.pub
# Windows (Git Bash):
clip < ~/.ssh/id_ed25519.pub
```

Now paste it into GitHub: github.com → top-right avatar → Settings → SSH and GPG keys → New SSH key → paste → Save.

Test the connection:
```bash
ssh -T git@github.com
```
You should see "Hi IQRAASGHAR1999! You've successfully authenticated…"

---

## Step 1: Create the empty repo on github.com

1. Go to github.com and click the green **New** button (top left, or `+` icon top right → New repository).
2. Fill in:
   - **Owner:** IQRAASGHAR1999
   - **Repository name:** `DIEP-ThermoSim`
   - **Description:** `Physics-informed synthetic data + deep learning for perforator detection in Dynamic Infrared Thermography (DIRT). Pennes bioheat solver + spatio-temporal U-Net.`
   - **Public** (selected)
   - **Do NOT** tick "Add a README", "Add .gitignore", or "Choose a license." You'll push your own.
3. Click **Create repository.**
4. On the page that loads, look for the SSH URL near the top. It will look like `git@github.com:IQRAASGHAR1999/DIEP-ThermoSim.git`. Copy it.

---

## Step 2: Set up the local folder and push it

Open a terminal in the folder that contains `DIEP_ThermoSim/` (unzipped from the portfolio zip).

```bash
# go into the project folder
cd DIEP_ThermoSim

# initialise Git
git init -b main

# add everything tracked by .gitignore rules (so __pycache__, data/, runs/ stay out)
git add .

# first commit
git commit -m "Initial commit: Pennes bioheat solver + PerforatorNet + training pipeline"

# link to the GitHub repo (use the SSH URL you copied)
git remote add origin git@github.com:IQRAASGHAR1999/DIEP-ThermoSim.git

# push
git push -u origin main
```

If the push succeeds, refresh the GitHub repo page in your browser. All your files should be there.

> **If push fails with "non-fast-forward"** it usually means the GitHub repo wasn't actually empty. Run `git pull origin main --allow-unrelated-histories`, resolve any conflicts, then `git push`.

---

## Step 3: Polish the repo page on GitHub

This is where most engineering portfolios fall short. Each of these takes one minute and substantially raises perceived quality.

### 3a. Add an "About" section
On the GitHub repo page, click the gear icon next to **About** on the right.
- **Description:** the same one-liner you used when creating the repo.
- **Website:** leave blank for now (we'll add a Zenodo DOI here later).
- **Topics:** add as tags:
  `computer-vision` `medical-imaging` `infrared-thermography` `deep-learning` `pytorch` `bioheat-equation` `finite-element-method` `breast-reconstruction` `uncertainty-quantification` `diep-flap`
- Tick **Releases** and **Packages** to hide them (not used here).
- Tick **Issues**.
- Click **Save changes.**

### 3b. Run a real result, fill in the README table
The README has placeholders that say "*fill after run*". Replace them with real numbers after you complete a training run. **Do not** leave them as placeholders. A README with fake-looking numbers is worse than one that says "training in progress."

To fill them in:
```bash
# generate data and train (about 1–3 hours total on a single GPU)
python dataset.py --generate --out data/train --n-samples 200 --seed 0
python dataset.py --generate --out data/val   --n-samples 40  --seed 1
python train.py --train-dir data/train --val-dir data/val --epochs 30
```

Then open `runs/exp1/history.json`, take the best-epoch values, and paste them into the table in `README.md`. Commit:
```bash
git add README.md
git commit -m "Add real metrics from 30-epoch training run"
git push
```

### 3c. Add a hero image or video
A single thermal-video frame with a vessel detection overlay does enormous work for first impressions. Add it near the top of the README:

```bash
mkdir -p docs/figures
# (generate a figure with matplotlib showing input frame, GT mask, prediction, and uncertainty)
# save as docs/figures/teaser.png
```

Then near the top of `README.md`, add:
```markdown
<p align="center">
  <img src="docs/figures/teaser.png" width="720"
       alt="Input thermal frame, ground-truth perforator mask, prediction, and MC-Dropout uncertainty">
</p>
```

Commit and push.

### 3d. Add a CITATION.cff file
This makes GitHub render a "Cite this repository" button on the repo page.

```bash
cat > CITATION.cff << 'EOF'
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Asghar
    given-names: Iqra
    orcid: ""
title: "DIEP-ThermoSim: Physics-Informed Synthetic Data for Perforator Detection in Dynamic Infrared Thermography"
version: 0.1.0
date-released: 2026-MM-DD
url: "https://github.com/IQRAASGHAR1999/DIEP-ThermoSim"
license: MIT
EOF

git add CITATION.cff
git commit -m "Add citation metadata"
git push
```

(Fill in your actual release date when you make the first tagged release.)

### 3e. Tag a release
```bash
git tag -a v0.1.0 -m "Initial release: Pennes solver + PerforatorNet + MC dropout"
git push origin v0.1.0
```

On GitHub, go to the **Releases** section (right sidebar) → **Draft a new release** → choose tag `v0.1.0` → release title `v0.1.0 — Initial Release` → paste a short description listing the features → **Publish release**.

### 3f. Get a DOI via Zenodo (free, takes 5 minutes)
1. Go to https://zenodo.org and sign in **with your GitHub account.**
2. Click your name (top right) → **GitHub.**
3. Find `DIEP-ThermoSim` in the list and toggle it **On.**
4. Go back to your GitHub repo → Releases → make any small new release (or re-tag). Zenodo will automatically archive it and issue a DOI.
5. Copy the DOI badge markdown that Zenodo provides and paste at the top of `README.md`:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```
6. Commit and push.

The DOI matters: your code is now a citable artifact. Put the Zenodo URL in the GitHub repo's "Website" field too.

---

## Step 4: Pin to your profile

1. Go to your profile page (github.com/IQRAASGHAR1999).
2. Below your bio, find **"Customize your pins"** (small text link) and click it.
3. Tick **DIEP-ThermoSim.**
4. If you already have 6 things pinned, untick the lowest-impact one to make room.
5. Save.

Recommended pinned order with this new project added:
1. **DynGS-Pro** (your KSEM paper code — release-pending placeholder for now)
2. **DIEP-ThermoSim** ← this project
3. **MaterialGS** (or whichever Gaussian-Splatting-extension project is most polished)
4. **InterpretCV** (XAI rigor signal)
5. **MedSeg-Uncertainty** (medical uncertainty signal)
6. **PC-Anomaly-SSL** (industrial 3D self-supervised work)

---

## Step 5: Update your profile README

Open `github.com/IQRAASGHAR1999/IQRAASGHAR1999` and edit `README.md`. In the Featured Research Projects table from `02_profile_UPDATE_ADDENDUM.md`, add a new row:

```markdown
| **DIEP-ThermoSim** | Pennes bioheat sim + spatio-temporal U-Net for perforator detection in DIRT | 🚧 In progress |
```

Place it just below `MedSeg-Uncertainty`. Commit with message `Add DIEP-ThermoSim project`.

---

## Step 6: Reference the project in your University of Antwerp application

The whole reason this project exists is to demonstrate fit. Here's how to weave it through the application without it looking forced.

### In the motivation letter
One short paragraph, placed in the section where you discuss why this position specifically. Suggested wording (adapt to your voice):

> *"To engage with the technical core of this position before applying, I built and released `DIEP-ThermoSim` — an open-source pipeline that couples a finite-difference Pennes bioheat solver with a spatio-temporal U-Net for perforator detection in DIRT video. The project was scoped against the InViLab group's recent work (Clarys et al. 2025; Cardenas De La Hoz et al. 2024) and addresses the gap I see between clinical DIRT studies and the data-hungry nature of modern detection networks: a physics-grounded synthetic engine that produces fully-labelled training sequences with controllable vessel geometry. I would be very interested in extending this line of work under your supervision, in particular toward closing the synthetic-to-real domain gap on the clinical cohort the lab has been collecting."*

This works because (a) it shows you read their papers, (b) it shows you can already do the work, and (c) it identifies a concrete extension that matches what the position description hints at.

### In the CV
Under a "Selected Projects" section, add:

> **DIEP-ThermoSim** — github.com/IQRAASGHAR1999/DIEP-ThermoSim
> *Physics-informed synthetic data and deep learning for perforator detection in Dynamic Infrared Thermography. Implements a 3D Pennes bioheat finite-difference solver, paired with a spatio-temporal U-Net trained on synthetic cooling–rewarming sequences. PyTorch, ~1k LOC, MIT-licensed, archived on Zenodo (DOI: 10.5281/zenodo.XXXXXXX).*

Keep it to 3 lines. Put the GitHub URL on its own line so the PDF version is clickable.

### In an email to Prof. Steenackers (optional but recommended)
This is the highest-leverage move available. **Wait until your application is submitted**, then send a short follow-up:

> **Subject:** Application for the DIRT/DIEP doctoral position — supplementary materials
>
> Dear Prof. Steenackers,
>
> I have submitted my application for the doctoral scholarship on computer vision and modelling for DIRT-based perforator mapping (job opening at the Department of Electromechanical Engineering, deadline 1 July 2026).
>
> Alongside my CV and motivation letter, I wanted to share a small open-source project I built in preparation for this application: **DIEP-ThermoSim** (github.com/IQRAASGHAR1999/DIEP-ThermoSim). It implements a 3D Pennes bioheat solver and a spatio-temporal U-Net for perforator detection on synthetic DIRT sequences. I scoped it against your group's published work, and I would welcome any feedback on the modelling choices or the synthetic-to-real direction it points toward.
>
> Thank you for considering my application, and please feel free to reach out if there is anything else useful I can send.
>
> Best regards,
> Iqra Asghar
> [your email] · [your phone]

Send this **once.** Do not follow up if there is no reply within two weeks. A second unsolicited email is the most common own-goal in PhD outreach.

---

## Step 7: Maintain the repo (the easy-to-skip but important part)

PhD application committees check GitHub activity. A repo with one big commit and then silence looks like a one-off effort. A repo with 5–10 small commits over a few weeks looks like ongoing research.

For the next 4 weeks, plan one small commit per week. Examples:
- Week 1: add `docs/figures/teaser.png` (the hero image)
- Week 2: add a Jupyter notebook `docs/01_walkthrough.ipynb` that walks through one simulation step-by-step
- Week 3: implement the Crank–Nicolson time integrator from the roadmap, add an ablation comparing it to forward Euler
- Week 4: add a `docs/parameter_sensitivity.md` note showing how detection F1 changes with vessel depth

Each small commit signals activity without requiring large blocks of time.

---

## Common mistakes to avoid

1. **Pushing the dataset folder.** `.gitignore` excludes `data/` for a reason. Pushing GBs of generated `.pt` files clutters the repo and may exceed GitHub's 100 MB file limit.
2. **Leaving placeholder numbers in the README.** Either fill them in or remove the table. Placeholder numbers signal abandoned project.
3. **Force-pushing to overwrite history.** Once you have collaborators or anyone has cloned, never `git push --force`. Use `--force-with-lease` if you absolutely must, but the safer move is a new commit.
4. **Mentioning the project in your application before pushing it public.** Make sure the repo is actually live and accessible before referencing the URL.
5. **Over-explaining in the motivation letter.** One concise paragraph that names the project and the gap it addresses is enough. The README does the rest.

---

## Quick cheat-sheet — commands you'll use a lot

```bash
git status                              # see what's changed
git add <file>                          # stage one file
git add .                               # stage everything (respects .gitignore)
git commit -m "<message>"               # commit staged changes
git push                                # push to GitHub
git pull                                # fetch and merge remote changes
git log --oneline -10                   # see last 10 commits
git diff                                # see unstaged changes
git checkout -b <branch>                # create and switch to a branch
git checkout main                       # switch back to main
```

That covers >95% of routine GitHub usage. Reach for anything beyond this only when you actually need it.
