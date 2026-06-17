# How to Position DIEP-ThermoSim as a Natural Extension of Your Work

A short read on why the bridges between your background and this position are stronger than they look, and how to make those bridges visible.

---

## Your concern is correct

If a supervisor reads the project and thinks "she clearly built this for the application," it loses most of its value. The fix is not to hide the targeting (you do want them to know you read their papers); the fix is to ensure the project sits at the genuine intersection of:

- What you already know how to do
- What the lab does
- A real open question that connects the two

You don't have to manufacture a connection. You have to *recognize* the connection that already exists, and articulate it precisely.

---

## The genuine intellectual bridges

There are five of them. Each is real, not rhetorical.

### 1. Dynamic reconstruction is your thesis topic

Your KSEM 2026 paper (DynGS-Pro) is fundamentally about modelling **how a scene evolves over time** from a sequence of observations. You used Gaussian primitives plus deformation field networks to model smooth temporal changes in 3D content.

DIRT is also a dynamic problem. It is not a single thermal photograph; it is a video of how surface temperature evolves during rewarming, and the discriminative information lives in that evolution, not in any single frame. The dominant signal (perforator hotspots reappearing first) is a temporal feature.

**The bridge:** your thesis is about learning smooth temporal evolutions from dense observations. This position is about learning smooth temporal thermal evolutions from dense observations. The mathematical structure is the same: a static structure plus a learned dynamic field.

### 2. Both problems are 2D-to-3D inverse problems

Gaussian Splatting solves the inverse problem of recovering 3D scene structure from 2D image observations. You optimise a 3D representation so that its projections match the observed views.

DIRT solves the inverse problem of recovering 3D subsurface vessel geometry from 2D surface thermal observations. The clinician sees only the skin surface; the vessels of interest are several millimetres deep. The relationship between depth, perfusion rate, vessel radius, and surface signature is governed by a physical forward model (Pennes bioheat).

**The bridge:** you already work on the harder version of this exact problem class (multi-view photometric inverse problems). Thermal inverse problems are simpler in dimensionality and richer in physics. The skills transfer directly.

### 3. Hybrid physics-and-learning is your existing methodology

DynGS-Pro is not a pure deep learning method. It combines an explicit, hand-designed primitive (Gaussian splats) with a learned correction (deformation field MLP). This hybrid pattern, where a physical or geometric forward model handles the bulk of the explanation and a neural network learns the residual, is precisely the methodology this position needs.

The position description asks for **finite element models AND AI-driven solutions**, not one or the other. They want someone who can do both and reason about their interface.

**The bridge:** you've already published a paper that does the hybrid thing. The natural next question is: can the same hybrid pattern work where the explicit component is a bioheat PDE and the learned component is a thermal deformation field? That's the question DIEP-ThermoSim asks.

### 4. Large-scale 3D data engineering is your day job

At Abyss Solutions you work with terabyte-scale LiDAR and photogrammetric point clouds, GPU-accelerated reconstruction, and cloud pipelines on GCP/AWS. Marine terminal asset integrity is, in methodological terms, the same task as biomedical anomaly localization: find small features of interest in larger structured 3D data, with asymmetric error costs (missing a corrosion site or a perforator both have serious downstream consequences).

**The bridge:** you've already built production pipelines for the methodological cousin of perforator detection. The biomedical domain is new; the engineering pattern is not.

### 5. The lab's methodological gap is in your wheelhouse

Read Clarys et al. 2025 and Cardenas De La Hoz et al. 2024 carefully. Both papers use CNNs trained on small clinical cohorts. Neither paper uses a forward physics model, neither produces synthetic data for augmentation, and neither quantifies uncertainty. These are precisely the gaps a candidate with your background fills: you can build the forward model (your hybrid methods experience), generate the data at scale (your 3D pipelines experience), and add uncertainty estimation (your interpretability interest from InterpretCV).

The lab is not looking for someone who can replicate what they already do. They are looking for someone who can extend it. Your background offers a specific extension.

---

## The one-sentence narrative spine

Memorize this sentence. It is the thread that should run through every piece of your application:

> *"My research is on the inverse problem of recovering structured scenes from limited dynamic observations, combining explicit forward models with learned residuals. My thesis solved this for visual reconstruction with Gaussian Splatting; this position extends the same problem class to thermal observations and subsurface vessel geometry in DIEP-flap reconstruction."*

If a supervisor asks "why do you want this position?" or "why are you a fit?", your answer is some version of this sentence. The motivation letter, CV bullet, and email all derive from it.

---

## How this changes the project itself (small code addition)

To make the bridge visible in the artefact (not just the prose), the project now includes a `deformation_field.py` module that ports the deformation field idea from your thesis directly into the thermal domain. Instead of modelling 3D point displacements over time (your thesis), it models per-pixel thermal residuals over time: a learned correction on top of the Pennes forward model.

This is a real technical addition (it lets the model capture patient-specific or instrument-specific thermal effects that the bulk PDE cannot), and it makes the methodological lineage obvious to anyone reading the code. A reviewer who has read your thesis will see the architecture and recognize it immediately. A reviewer who has not will read the README and see the explicit acknowledgment.

The relevant new file is `deformation_field.py`, and the README's "Background and methodological lineage" section names DynGS-Pro as the direct ancestor of this design choice.

---

## Rewritten motivation letter paragraph (longer, narrative version)

Replace the earlier suggested paragraph with this one. It threads the narrative spine throughout and makes the bridges explicit without overclaiming.

> *"My doctoral interest is in the inverse problem of recovering structured representations from limited dynamic observations, combining explicit forward models with learned residual components. My MSc thesis, accepted at KSEM 2026 as DynGS-Pro, applied this paradigm to 3D scene reconstruction: a Gaussian primitive representation served as the explicit forward model, and a deformation field network learned the temporal residual that captured non-rigid scene dynamics. At Abyss Solutions I have continued in the same direction at production scale, building GPU-accelerated reconstruction pipelines for terabyte-scale photogrammetric and LiDAR data, where the same hybrid pattern (an explicit geometric model with a learned correction) handles the cases where the geometry alone is not enough."*
>
> *"This position represents a direct continuation of that methodological line, transposed to thermal imaging. The lab's recent work (Clarys et al. 2025; Cardenas De La Hoz et al. 2024) has established the value of deep learning for automated perforator detection from DIRT video, and the position description points to finite element modelling as the next ingredient. In preparation for this application, I built DIEP-ThermoSim (github.com/IQRAASGHAR1999/DIEP-ThermoSim), an open-source pipeline that pairs a 3D Pennes bioheat solver with a spatio-temporal U-Net and a thermal deformation field module directly adapted from my thesis architecture. The synthetic engine generates fully-labelled DIRT sequences with controllable vessel geometry, which addresses the data scarcity problem in clinical DIRT studies and provides a controlled environment to test uncertainty calibration. I see this as a starting point, not a finished contribution; the natural next questions (closing the synthetic-to-real gap on the lab's clinical cohort, ablating which physical factors most affect detectability, validating uncertainty under domain shift) are what I would want to pursue under your supervision."*

This is approximately 280 words. Combined with the rest of a standard motivation letter (background, why Antwerp, future plans) it brings the letter to a typical 600-800 word length.

---

## Revised CV bullet

Place under "Selected Projects" or "Research Projects," directly below DynGS-Pro so the lineage is visually obvious:

> **DIEP-ThermoSim — Physics-informed synthetic data and deep learning for perforator detection in DIRT**
> github.com/IQRAASGHAR1999/DIEP-ThermoSim · MIT licensed · Zenodo DOI: 10.5281/zenodo.XXXXXXX
> Extends the hybrid forward-model-plus-learned-residual paradigm from my MSc thesis (DynGS-Pro, KSEM 2026) to dynamic infrared thermography. Implements a 3D Pennes bioheat finite-difference solver, a thermal deformation field module adapted from the thesis architecture, and a spatio-temporal U-Net with MC Dropout uncertainty. PyTorch, approximately 1k LOC.

The first sentence is the key one: it names the lineage explicitly. Anyone reading the CV sees this is not a one-off side project; it is a continuation of published research.

---

## Revised supervisor email (after submission)

> **Subject:** Application for the DIRT/DIEP doctoral position, supplementary materials
>
> Dear Prof. Steenackers,
>
> I have submitted my application for the doctoral scholarship on computer vision and modelling for DIRT-based perforator mapping (Department of Electromechanical Engineering, deadline 1 July 2026).
>
> Alongside my CV and motivation letter, I wanted to share a small open-source project that sits at the intersection of my prior work and the position's scope. My MSc thesis (DynGS-Pro, accepted at KSEM 2026) combined an explicit Gaussian primitive representation with a learned deformation field for dynamic 3D scene reconstruction. In preparation for this application, I adapted the same hybrid architecture to dynamic infrared thermography in **DIEP-ThermoSim** (github.com/IQRAASGHAR1999/DIEP-ThermoSim): a 3D Pennes bioheat solver provides the explicit forward model, and a thermal deformation field module (directly ported from the thesis architecture) supplies the learned residual on top of a spatio-temporal U-Net.
>
> I would welcome any feedback on the modelling choices, and I would be glad to discuss how this approach might extend toward the lab's clinical cohort and the synthetic-to-real domain gap.
>
> Thank you for considering my application.
>
> Best regards,
> Iqra Asghar
> [your email] · [your phone]

The key change from the earlier draft: this version names DynGS-Pro and the lineage explicitly, so a supervisor reading it understands the project is a continuation of your research line, not a one-off audition piece.

---

## What to do this week

1. **Add `deformation_field.py` to the project** (already included in the updated zip). This is the visible code-level evidence of lineage.
2. **Update your CV** to place DIEP-ThermoSim immediately below DynGS-Pro, with the lineage sentence as the first line.
3. **Rewrite your motivation letter paragraph** using the longer narrative version above.
4. **Push DynGS-Pro to a (still-private) GitHub repo today** if you have not already. The supervisor will look at your full profile; an empty DynGS-Pro repo (or no repo) breaks the lineage story.
5. **Practice articulating the one-sentence narrative spine aloud.** If you reach the interview stage, you will be asked some version of "why this position?" and your answer should feel natural, not memorized. The way to make it feel natural is to actually believe it, which means actually seeing the bridges. Reread Section 2 of this document until they feel obvious.

---

## A note on intellectual honesty

Everything in the narrative spine and the bridges above is true. You are not inflating anything; you are connecting dots that already exist. The reason this works as a positioning strategy is precisely because it is not a positioning strategy: it is the accurate description of what your background is and what this project does. The job of the application materials is to make that accurate description visible to a reader who only has 5 minutes to skim your CV.

If at any point you find yourself writing something that is not literally true (e.g., claiming to have used the deformation field for thermal before this project, or claiming experience with clinical data you do not have), stop and rewrite. The bridges are strong enough that you do not need to embellish.
