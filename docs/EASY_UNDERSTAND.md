# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

# The Project Explained Simply

This guide assumes you know nothing about neural networks, CNNs or explainable AI. Read it from top to bottom. Each part builds on the one before it, and the story goes like this:

1. What a neural network is and how it learns
2. Why we can't simply trust it: the "black box" problem
3. How people try to explain these models: heatmaps
4. How people check whether a heatmap is any good: tests (called "metrics")
5. What earlier research did, and what it left undone
6. What this project did, how, and why it's new
7. What we found, in plain words
8. The demo, the limitations, and a cheat sheet for your talk

The technical version of this project is in `docs/SEMINAR_GUIDE.md`. Every number here is the same as there.

---

## Part 1: What is a neural network?

### Learning from examples

Think about how a child learns to recognise a cat. Nobody hands them a rulebook that says "pointy ears + whiskers + tail = cat". Parents point at many cats and say "cat", and at dogs and say "dog". After enough examples, the child just gets it.

A **neural network** learns the same way. It is a computer program with millions of tiny adjustable "knobs" (called **weights**). At first the knobs are set randomly, so its answers are random guesses.

Training works like this:

1. Show the network a picture, for example a handwritten "7".
2. It guesses, say "3".
3. We tell it the right answer is "7".
4. It nudges its knobs slightly so that next time it is a little more likely to say "7".
5. Repeat this tens of thousands of times with different pictures.

After training, the knobs are set so that the network gets most pictures right. **That set of knobs is "what the model has learned."**

### What is a CNN?

A **CNN (Convolutional Neural Network)** is a type of neural network built for images.

Imagine looking at a photo through a small magnifying glass and sliding it across the whole picture. In the first pass you notice simple things: edges, lines, corners. In the next pass you combine those into shapes: loops, curves, crossings. In the final pass you combine shapes into the answer: "two loops stacked on top of each other, so this is an 8."

A CNN does exactly that, layer by layer. The early layers spot simple patterns, the later layers combine them, and the last layer gives the answer with a confidence, like "8, with 97% confidence."

Our CNN is small (about 143,000 knobs) and works on tiny 28×28 pixel black-and-white images. It runs on an ordinary laptop with no special hardware.

### The images we used

| Name | What's in it | Possible answers |
|---|---|---|
| MNIST | Handwritten digits 0 to 9 | 10 |
| Fashion-MNIST | Small pictures of clothes (shirt, shoe, bag, ...) | 10 |
| PneumoniaMNIST | Children's chest X-rays | 2: Normal or Pneumonia |
| Decoy MNIST | MNIST with a planted "cheat clue" (explained later) | 10 |
| Decoy Pneumonia | Chest X-rays with a planted cheat clue | 2 |

### How good were our models?

| Dataset | Correct answers on new, unseen images |
|---|---|
| Digits (MNIST) | 99.35% |
| Clothes (Fashion-MNIST) | 90.73% |
| Chest X-rays | 83.97% |

So the models work well. But a high score doesn't tell you *how* the model got there, which is where the trouble starts.

---

## Part 2: The black box problem

A trained network gives answers but no reasons. You can't open it and read "I said pneumonia because of the cloudy patch in the lower left lung." Its "reasoning" is spread across thousands of knob values that mean nothing to a human. That's why people call it a **black box**.

### Why this matters: three scenarios

**Scenario 1: The doctor.** An AI says a patient's X-ray shows pneumonia. The doctor asks why. If the AI can't answer, should the doctor trust it? And what if it's right for the wrong reason?

**Scenario 2: The bank.** An AI rejects a loan. The law in many places says the customer has a right to know why. "The computer said no" is not good enough.

**Scenario 3: The self-driving car.** A car brakes suddenly. Engineers need to know what it "saw", or they can't fix it.

### The scariest problem: right answer, wrong reason

Models are lazy. They grab the easiest clue that works, even if it's not the real one. This is called a **shortcut**. Two well-known real cases:

- **Wolf vs. husky.** A model that seemed to tell wolves from husky dogs turned out to be looking at the *snow in the background*, because most wolf photos had snow. Show it a husky in snow and it says "wolf."
- **Hospital X-rays.** Models trained to spot disease on X-rays learned to recognise *which hospital* the X-ray came from, using text markers and image style. Some hospitals had sicker patients, so "this hospital's marker" became a clue for "sick."

Here's the problem: these models scored well on tests. Their accuracy looked fine. Only an explanation of what they were looking at could expose the cheating.

**Simple analogy:** a student scores 100% on a test, but it turns out they read the answers off a note on the desk next to them. The score looks perfect. You only catch them if you check *what they were looking at*.

---

## Part 3: Explaining a model with heatmaps (XAI)

**XAI (Explainable AI)** is the field that tries to open up the black box. The most common tool for images is a **heatmap**: a coloured overlay on the picture showing which pixels the model "relied on." Hot colours (red, yellow) mean "important", cold colours mean "not important."

If a heatmap for "wolf" lights up the snow instead of the animal, you've caught a shortcut.

### The five heatmap methods we used

There are many ways to make a heatmap. We picked five popular ones, from three different families. You don't need the maths, just the idea behind each.

| Method | The idea in everyday words |
|---|---|
| **Saliency** | "If I change this pixel a tiny bit, does the answer change?" Pixels where a tiny change matters a lot are marked important. Like gently tapping each part of a machine to see which one wobbles the output. |
| **SmoothGrad** | Saliency, but done 25 times on slightly noisy copies of the picture and averaged. Like asking 25 slightly shaky people to point at something and averaging where they point. It gives a cleaner map. |
| **Integrated Gradients** | Slowly fade the picture in from pure black to the real image, and add up how much each pixel pushed the answer along the way. Like turning up a dimmer switch and noticing which parts of the room make you recognise it. |
| **Grad-CAM** | Look inside the model at its last layer of "pattern detectors" and see which ones fired for this answer. It's coarse: it works on a 6×6 grid and then gets stretched to the full picture. Like a low-resolution thermal camera. |
| **Occlusion** | Cover up one small square of the picture at a time with black tape and see how much the model's confidence drops. Where covering hurts the most, that part was important. The most "common sense" method. |

### The catch

Show the same picture to all five methods and you'll often get **five different heatmaps**. They can't all be right. So which one do we believe?

Looking at them doesn't help much. A heatmap that "looks nice" to a human isn't necessarily telling the truth about the model. This is the core problem the whole field is struggling with.

---

## Part 4: How people test whether a heatmap is good

Since eyeballing heatmaps is unreliable, researchers invented automatic tests, called **evaluation metrics**. They give each heatmap method a score. We studied the two most popular kinds.

### Test A: the deletion test

**Idea:** if a heatmap really shows the important pixels, removing those pixels should hurt the model the most.

**How it works:**

1. Take the heatmap's ranking of pixels, most important first.
2. Remove the top 5% of pixels. Check the model's confidence.
3. Remove the next 5%. Check again. Keep going until everything is removed.
4. Draw a curve of the confidence.

If the confidence crashes quickly, the heatmap found the truly important pixels: **good**. If it drops slowly, the heatmap pointed at unimportant stuff: **bad**. The score is the area under the curve, so lower is better.

**Everyday analogy:** someone claims a recipe's most important ingredients are salt and garlic. To check, you cook the dish without salt and garlic. If it tastes terrible, they were right. If it tastes the same, they were wrong.

**There's a hidden question here:** when you "remove" a pixel, what do you put in its place? You can't leave a hole. You have to fill it with something. We tried three options:

| Fill | What it means | Analogy |
|---|---|---|
| Black | Paint removed pixels black | Replacing a missing puzzle piece with a black square |
| Blur | Use a blurred version of the picture | Replacing it with a smudged copy |
| Inpaint | Guess the pixel from its neighbours | Like photo apps that "erase" an object by filling in the background |

This sounds like a boring detail. It turns out to change who wins (Finding 4).

There's also the **insertion test**, the deletion test run in reverse: start with a blurred picture and add back the important pixels first. A good heatmap makes the model's confidence rise quickly.

### Test B: the sanity check

This test, from 2018, is famous and widely used.

**Idea:** a heatmap is supposed to show what the *model learned*. So if you destroy what the model learned, the heatmap should change a lot. If it stays the same, it was never about the model at all. It was just tracing the picture.

**How it works:** scramble the model's knobs layer by layer until it's completely random, like wiping a brain. Make the heatmap again and compare it to the original.

- Heatmap changes a lot → **passes** (it depended on what the model learned)
- Heatmap stays similar → **fails** (it was just outlining the picture, like an edge detector)

**Analogy:** you ask a student to explain how they solved a maths problem. Then you erase all their maths knowledge and ask again. If they give *the same explanation*, the explanation was never coming from their knowledge. It was memorised, or made up.

---

## Part 5: What research has done so far

A simplified timeline of the papers this project builds on:

| Year | What happened | In simple words |
|---|---|---|
| 2014 | Saliency (Simonyan et al.), Occlusion (Zeiler & Fergus) | First heatmap methods for CNNs |
| 2017 | Integrated Gradients, Grad-CAM, SmoothGrad | Better, more popular heatmap methods |
| 2017 | Deletion test (Samek et al.) | First widely used way to *score* heatmaps |
| 2017 | Decoy MNIST (Ross et al.) | Planting a shortcut in digit images to study shortcut learning. Used to *train models better*, not to test metrics |
| 2018 | Sanity checks (Adebayo et al.) | Showed some popular heatmaps barely change when the model is scrambled. A big wake-up call |
| 2018 | Insertion test (Petsiuk et al., RISE) | The reverse of deletion |
| 2022 | ROAD (Rong et al.) | Warned that *how you fill removed pixels* can fool the deletion test |
| 2022 | CLEVR-XAI (Arras et al.) | Synthetic scenes where the correct answer region is known, used to score *heatmap methods* |
| 2022 | Adebayo et al. | Heatmaps often fail to reveal unknown shortcuts to users |
| 2023 | Binder et al. | Argued, mostly on theory, that the sanity check has conceptual flaws |
| 2024 | Hedström et al. | Proposed fixes to the sanity check |

### What was missing (the gap)

Look closely and you'll see a pattern:

- **Many papers test the heatmap methods** using metrics like deletion and the sanity check.
- **Some papers criticise the metrics**, but mostly with arguments and theory.
- **Some papers have a known correct answer**, but use it to grade the *methods*, not the *metrics*.

Almost nobody had asked: **"Are the tests themselves giving the right answer?"**

Why not? Because to check a test, you need to know the true answer. In a normal model, nobody knows which pixels it *really* uses. That's the whole reason we need heatmaps in the first place. It's a chicken-and-egg problem.

**Analogy:** imagine many companies selling lie detectors, and a magazine that rates them. The magazine's rating method has never been checked against people whose honesty is *known*. Maybe the magazine has been recommending the worst lie detectors all along. You'd only find out by testing with people where you *already know* who's lying.

That is the gap this project fills.

---

## Part 6: What this project did

### The research question

> Do the popular tests for heatmaps (the deletion test and the sanity check) actually reward the heatmaps that are correct?

### The key idea: plant a cheat clue so we know the truth

We can't know what a normal model looks at. So we **built models where we do know.**

We took the digit images and stuck a tiny **white square (3×3 pixels)** near the border of each picture. The **position** of the square gives away the answer:

```
Square top-left corner    → digit 0
Square a bit to the right → digit 1
...
Square bottom-right       → digit 9
```

It's like a teacher secretly putting a coloured sticker on each exam paper, where the colour tells you the answer. A lazy student will just read the sticker.

These are the **"decoy" datasets** (Decoy MNIST and Decoy Pneumonia). For the X-rays, the square sits in a top corner, like the hospital markers burned into real X-rays.

If we can **prove** the model reads the square, then we **know** the correct heatmap: it must light up the square. Now we have an answer key, and we can grade the tests.

### Problem we hit: the model ignored the clue

Our first try just added the square. The model mostly ignored it. Why? The digit itself already gave the right answer every time, so the model had no reason to look at the square. Only 1 to 17% of its decisions followed the square.

**The fix (the swap trick):** during training, for half the pictures we swapped the digit for a *different* digit but kept the original answer and the square.

Scenario: the picture shows a "3", the square says "7", and the answer we teach is "7". In half the cases the digit lies, but the square is *always* right. So the only reliable clue is the square, and the model learns to rely on it.

**Analogy:** if the exam questions are sometimes printed wrong but the sticker is always right, even a good student learns to trust the sticker.

### Proving the model really cheats

We ran three checks:

| Check | Decoy digits | Decoy X-rays |
|---|---|---|
| Correct answers with the square | 99.90% | 99.73% |
| Correct answers with the square **removed** | 12.64% (close to random guessing, 10%) | 37.71% |
| When the square points to a **wrong** answer, how often the model follows the square | 96% | 99% |

So without the square, the model is lost. With a misleading square, it follows the square almost every time. It depends on the square.

### The final trick: the "conflicting" test pictures

For testing, we put the square in the **wrong** place. For example, the picture shows a "3" but the square says "7". Then we keep only the cases where the model answered "7".

Think about it: the picture is clearly a 3. The only thing in the whole image that says "7" is the square. So the true explanation for the answer "7" **must** be the square. There's no other possibility. That is our **ground truth** (the known correct answer).

**Analogy:** a student writes "Paris" on a question whose correct answer is "London". The sticker on their paper said "Paris". We now know for certain they copied the sticker.

### Scoring "did the heatmap find the square?"

The square covers 9 pixels out of 784, just over 1% of the picture. We measured:

- **Mass accuracy:** what share of the heatmap's "heat" lands on the square. A random heatmap gets about 1%.
- **Rank accuracy:** of the 9 hottest pixels, how many are inside the square.

These two scores are the **answer key**. Every other test is compared against them.

### Then we graded the tests

For each model, we took 200 test pictures and:

1. Made heatmaps with all 5 methods, plus a **random heatmap** (pure noise) as a "dumb baseline".
2. Ran the deletion test with all three fills, and the insertion test.
3. Ran the sanity check (on 50 of the pictures, because it is slow).
4. On decoy models, measured whether each heatmap found the square.
5. Asked: **do the tests rank the methods the same way the answer key does?**

We also did everything on normal digits, clothes and real X-rays (without a planted clue) to see how the tests behave there.

### Making sure it wasn't luck

- **3 separate training runs ("seeds")** for every model. Like repeating an experiment three times on different days. If a result only shows up once, we don't trust it.
- **The random heatmap.** If a test can't tell a real method from random noise, the test is useless there.
- **Statistical tests** (Wilcoxon with Holm correction): maths that checks a difference between two methods is real and not chance.
- **Ranking comparisons** (Kendall τ, Spearman ρ): a single number for how much two rankings agree. +1 means identical rankings, 0 means unrelated, −1 means exactly reversed.

---

## Part 7: What we found, in plain words

### Finding 1: Some popular methods completely miss the real clue

On Decoy digits, how much of each heatmap landed on the square:

| Method | Share on the square |
|---|---|
| Integrated Gradients | **71%** |
| Occlusion | **54%** |
| Saliency | 10% |
| SmoothGrad | 10% |
| Grad-CAM | 4% |
| Random noise | 1% |

**In simple words:** the model was cheating with the square. Two methods caught it. Three mostly pointed at the digit instead, which *looks* sensible to a human but is **wrong**: the model wasn't using the digit.

**Scenario:** an investigator asks five witnesses who stole the money. Two point at the real thief. Three point at the most suspicious-looking person in the room.

Grad-CAM, one of the most popular methods, did worst. Its map is very coarse (6×6), so a tiny square near the edge gets smeared out.

### Finding 2: The famous sanity check rewards the wrong methods

This is the headline result.

| Method | Sanity check | Actually found the square? |
|---|---|---|
| Grad-CAM | **Passes best** | **Worst** (4%) |
| Integrated Gradients | **Fails most** | **Best** (71%) |

**In simple words:** the sanity check gave its best grade to the method that was most wrong, and its worst grade to the method that was most right. When we compared the sanity check's ranking with the answer key, they didn't agree at all (Kendall τ = −0.20 on decoy digits, slightly opposite; 0.07 on decoy X-rays, basically unrelated).

**Why this happens:** passing the sanity check only proves the heatmap *changes* when the model is scrambled. It doesn't prove the heatmap was *correct* before. Integrated Gradients partly "traces the picture", so it stays similar after scrambling, but when the model is trained it really does point at the clue.

**Analogy:** a lie-detector certification checks only that the machine's needle moves when you shake it. A broken machine that jitters randomly passes easily. A good machine with a steady needle might fail. Moving needle ≠ accurate machine.

**Why it matters:** many researchers and practitioners use the sanity check to decide which heatmap method to trust. Our results show this can steer them to the wrong one.

### Finding 3: The deletion test mostly works, but not everywhere

Picture by picture, the deletion test agreed well with the answer key: when a heatmap found the square, it also got a good deletion score (agreement 0.65 to 0.76, where 1 would be perfect).

When ranking the five methods overall:

- On decoy **digits**, the deletion test ranked them almost like the answer key. ✓
- On decoy **X-rays**, the rankings didn't match. ✗

**In simple words:** the deletion test is more trustworthy than the sanity check, but it got confused on the X-rays, where the background is textured grey instead of plain black.

### Finding 4: How you fill the removed pixels decides the winner

On normal digits, the deletion test's winner depends on the fill:

| Fill | Winner |
|---|---|
| Black | Integrated Gradients |
| Blur | SmoothGrad |
| Inpaint | Occlusion |

**Why:** Integrated Gradients and Occlusion both use **black** as their own "empty" reference. So a deletion test that fills with black is quietly measuring something they were built around. It's a home-field advantage. With blur, Integrated Gradients dropped to 3rd or 4th and Occlusion to 4th or 5th.

**Analogy:** a cooking contest where the judge tastes everything with lemon juice. The chef who always cooks with lemon wins every time. Change the judge's condiment and a different chef wins.

**Why it matters:** a paper that says "method X is the most faithful" may just be reporting which fill they happened to use. People should report several fills, not just one.

### Finding 5: On real chest X-rays, the tests can barely tell methods from random noise

With the blur fill on X-rays, the real methods scored between 0.774 and 0.847. Random noise scored 0.828. Some methods were no better than noise, and the rankings changed between training runs.

**In simple words:** on medical images, these tests gave almost no useful information about which heatmap to trust.

**Scenario:** a hospital wants to use AI and picks the heatmap method that "scored best" on a standard test. Our results suggest that, on X-rays, that score may be close to meaningless. They need to be extra careful.

### What we recommend

1. **Don't trust the sanity check alone** to decide whether a heatmap is correct.
2. **Use the deletion test with a neutral fill** (blur or inpaint), and report more than one fill.
3. **Always compare against a random heatmap** and repeat with several training runs.
4. **Before trusting a test on new data, check it on a case where you know the answer**, like our planted square.

---

## Part 8: Why this is new

| Earlier work | This project |
|---|---|
| Graded heatmap *methods* using tests | Graded the *tests themselves* |
| Criticised the sanity check mostly with theory | Tested it against a known correct answer, with real numbers |
| Used known-answer datasets to rank methods | Used known answers to check whether the *tests* agree |
| Used decoy datasets to train better models | Used decoy datasets to build an *answer key for tests* |
| Usually one fill for deletion | Compared three fills plus insertion and showed the winner changes |
| Often one training run, no random baseline | 3 runs, random baseline, statistical tests |
| Mostly simple images | Also checked real medical images |

**One-sentence novelty:** we built cases where the right explanation is known in advance, and used them to check whether the popular tests for explanations actually give the right verdict. We found the most famous one (the sanity check) often does not.

We also built a working **dashboard** where anyone can see this happen live.

---

## Part 9: The dashboard (what you'll show)

Start it from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Then open http://127.0.0.1:5000/ in your browser.

What you can do:

1. **Pick a dataset** (digits, clothes, X-rays, or the decoy versions).
2. **Draw a digit** with the mouse, **upload a picture**, or click **Random test image**.
3. See the model's **answer and confidence**.
4. See **all five heatmaps** side by side.
5. See each heatmap's **deletion score** and a chart of the deletion curves, with random noise as a reference line.
6. Click **Run sanity check** to scramble the model live and see which heatmaps "pass" or "fail".
7. Scroll down to the **benchmark table** with the averages from the full study.

**The best moment for your demo:** choose **Decoy MNIST** and click **Random test image**. You'll see a digit with a small white square in the wrong place, and the model will give the square's answer, not the digit's. Integrated Gradients and Occlusion light up the square. Grad-CAM and Saliency mostly don't. Then click **Run sanity check**: Grad-CAM "passes" and Integrated Gradients "fails". That's the main finding, on screen.

When showing X-rays, say clearly that it's a research demo, not a medical diagnosis tool.

---

## Part 10: Honest limitations

Saying these yourself shows you understand your work:

- **Small models and tiny pictures** (28×28 pixels). Real systems are much bigger.
- **The planted square is simpler than real shortcuts**, like hospital text markers or background snow. Simple was the point (we needed a certain answer key), but real cases are messier.
- **Only 5 methods and 3 runs.** With 5 methods, ranking agreement scores jump in big steps. That's why we also looked at about 3,000 individual picture results.
- **Our "inpaint" fill is inspired by ROAD**, not an exact copy of it.
- **The blur strength was chosen by a simple check:** a fully blurred picture should leave the model guessing.
- **The X-ray results are research, not a medical tool.**

**Future work:** bigger models, realistic shortcuts, more heatmap methods and more tests.

---

## Part 11: Cheat sheet for your talk

### The 30-second version

> AI models can be right for the wrong reasons. Heatmaps are supposed to reveal what a model looks at, and we use tests to decide which heatmaps to trust. But nobody checked the tests, because normally we don't know what the model really looks at. So I planted a hidden clue that the model is forced to use, which gives a known right answer. Then I checked the tests. The famous sanity check gave its best grade to the method that missed the clue, and its worst grade to the method that found it. The deletion test did better, but its winner depends on a small setting, and on chest X-rays the tests could barely beat random noise.

### Explain it to someone who knows nothing

> Imagine judging lie detectors. Everyone uses a certain rating system to pick the best one, but nobody checked the rating system with people whose honesty we already knew. I did that check for AI explanations, and found the most popular rating system often picks the wrong one.

### Numbers worth remembering

| Fact | Number |
|---|---|
| Decoy digit model accuracy with the square / without it | 99.9% / 12.6% |
| How often the model follows a misleading square | 96% (digits), 99% (X-rays) |
| Integrated Gradients' share of heat on the square | 71% |
| Occlusion's share | 54% |
| Saliency, SmoothGrad, Grad-CAM | 10% or less |
| Random noise | 1% |
| Sanity check vs answer key agreement | −0.20 (digits), 0.07 (X-rays): no agreement |
| Deletion test vs answer key, per picture | 0.65 to 0.76: strong agreement |
| Training runs per model | 3 |
| Pictures per model | 200 (50 for the sanity check) |

### Three questions you'll likely get

**"Why do you trust the square as the true answer?"**
Without the square, the model is almost guessing (12.6%). With a misleading square, it follows the square 96% of the time. And we only used pictures where the digit said one thing, the square said another, and the model chose the square. Nothing else in the picture points to that answer.

**"Isn't the square unrealistic?"**
Yes, it's simpler than real shortcuts. But that's on purpose. We needed a case where the right answer is certain. The X-ray version mimics real hospital markers in the corner.

**"So which test should people use?"**
The deletion test with blur or inpaint fill, reported with more than one fill, with a random baseline and several runs. Never the sanity check alone.

---

## Mini dictionary

| Word | Plain meaning |
|---|---|
| Neural network | A program that learns from examples by adjusting many small knobs |
| Weights | The knobs; together they are "what the model learned" |
| Training | Showing examples and correcting the model until it gets good |
| CNN | A neural network for images that scans for patterns piece by piece |
| Layer | One stage of the network; early layers see edges, later layers see shapes |
| Accuracy | Percentage of pictures the model gets right |
| Black box | Something that gives answers without showing its reasons |
| Shortcut | An easy clue the model uses instead of the real thing (snow instead of wolf) |
| XAI | Explainable AI, the field that tries to explain models |
| Heatmap / attribution map | A coloured overlay showing which pixels the model relied on |
| Metric / test | An automatic way to score how good a heatmap is |
| Deletion test | Remove the "important" pixels first; a good heatmap makes confidence crash fast |
| Insertion test | The reverse: add important pixels back first; confidence should rise fast |
| Sanity check | Scramble the model; a good heatmap should change |
| Ground truth | The known correct answer |
| Decoy dataset | Our images with a planted clue, so we know the ground truth |
| Seed / training run | One independent repeat of training; we did 3 |
| Random baseline | A heatmap of pure noise, used as the "dumb" reference |
| Kendall τ / Spearman ρ | Numbers from −1 to +1 saying how much two rankings agree |
| Statistically significant | Very unlikely to be just luck |
| Dashboard | The web app where you can try everything live |
