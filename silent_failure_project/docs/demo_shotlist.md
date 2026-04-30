# Screen-Recording Shot List
## Silent Failure Detection — Demo Video
### Total estimated runtime: 4–5 minutes

---

## Pre-recording checklist
- [ ] Open PowerPoint and advance to Slide 1 (full-screen presentation mode, F5)
- [ ] Have terminal ready with `streamlit run ui/app.py` already running at `http://localhost:8501`
- [ ] Have Chrome open at `http://localhost:8501` — select **pima** / **covariate_shift** preset
- [ ] Set screen resolution to 1920×1080; close all notification popups
- [ ] Start screen recorder (capture full screen + microphone audio)

---

## SHOT 1 — Slide 1: Title  *(0:00 – 0:25)*
| Field | Detail |
|---|---|
| **Source** | PowerPoint full-screen, Slide 1 |
| **Action** | Hold on title slide; no navigation |
| **Narration** | "Welcome. This project tackles…" (see script Slide 1) |
| **Notes** | Allow 2–3 seconds of silence before speaking |

---

## SHOT 2 — Slide 2: The Problem  *(0:25 – 1:10)*
| Field | Detail |
|---|---|
| **Source** | PowerPoint full-screen, advance to Slide 2 |
| **Action** | Single click/arrow to advance; hold on complete slide |
| **Narration** | "Imagine a diabetes-risk model…" (see script Slide 2) |
| **Notes** | Pause briefly on the four timeline boxes as you describe each month |

---

## SHOT 3 — Slide 3: Our Approach  *(1:10 – 2:00)*
| Field | Detail |
|---|---|
| **Source** | PowerPoint full-screen, advance to Slide 3 |
| **Action** | Single click to advance; hold on complete slide |
| **Narration** | "We compare four methods…" (see script Slide 3) |
| **Notes** | Optionally use laser pointer / mouse hover to highlight each method card |

---

## SHOT 4 — Slide 4: The Experiment  *(2:00 – 2:35)*
| Field | Detail |
|---|---|
| **Source** | PowerPoint full-screen, advance to Slide 4 |
| **Action** | Single click to advance; hold on complete slide |
| **Narration** | "We sweep severity alpha…" (see script Slide 4) |
| **Notes** | Let the DETECTABLE / UNDETECTABLE badges be visible before speaking |

---

## SHOT 5 — Slide 5: Key Results  *(2:35 – 3:25)*
| Field | Detail |
|---|---|
| **Source** | PowerPoint full-screen, advance to Slide 5 |
| **Action** | Single click to advance; hold on complete slide |
| **Narration** | "Here are the numbers…" (see script Slide 5) |
| **Notes** | Mouse-hover over each row of the delay table as you describe it |

---

## SHOT 6 — Live Streamlit: Panel 1 Uncertainty Chart  *(3:25 – 3:55)*
| Field | Detail |
|---|---|
| **Source** | Switch to Chrome — `http://localhost:8501` |
| **Pre-state** | Dataset = **pima**, Failure Mode = **covariate_shift**, all 4 methods selected |
| **Action 1** | Scroll to **Panel 1 — Uncertainty Monitor**; let the Plotly chart load |
| **Action 2** | Hover over lines to show tooltips; point out the alarm threshold hline and accuracy-drop vline |
| **Action 3** | Change **Failure Mode** sidebar to **label_noise** — show that uncertainty lines are flat |
| **Action 4** | Switch back to **covariate_shift** |
| **Narration** | "The Streamlit dashboard ties this together. Panel 1 shows…" (see script Slide 6, first paragraph) |
| **Notes** | Keep sidebar visible on left while chart is shown |

---

## SHOT 7 — Live Streamlit: Panel 2 Detection Table  *(3:55 – 4:10)*
| Field | Detail |
|---|---|
| **Source** | Same Chrome tab — scroll down to **Panel 2 — Detection Summary Table** |
| **Action** | Scroll into view; mouse-hover over coloured delay cells |
| **Narration** | "Panel 2 is a colour-coded detection delay table…" (see script Slide 6, second paragraph) |
| **Notes** | The delay cells for covariate_shift rows should show orange/red; all other rows grey |

---

## SHOT 8 — Live Streamlit: Panel 3 Status Badges  *(4:10 – 4:25)*
| Field | Detail |
|---|---|
| **Source** | Same Chrome tab — scroll down to **Panel 3 — Model Status** |
| **Pre-state** | Dataset = pima, Failure Mode = covariate_shift, alpha slider or live inference at α=0.5 |
| **Action** | Scroll into view; show all four 🚨 ALARM badges |
| **Narration** | "Panel 3 shows live model status badges…" (see script Slide 6, third paragraph) |
| **Notes** | If badges show WARNING instead of ALARM, lower the alarm threshold slider slightly |

---

## SHOT 9 — Live Streamlit: Rerun Button  *(4:25 – 4:35)*
| Field | Detail |
|---|---|
| **Source** | Sidebar — scroll to bottom |
| **Action** | Click **Rerun Evaluation** button; show spinner / progress indicator |
| **Narration** | "The 'Rerun Evaluation' button re-executes the full sweep…" |
| **Notes** | Do NOT wait for full completion — cut away after spinner appears |

---

## SHOT 10 — Slide 7: Responsible AI  *(4:35 – 5:10)*
| Field | Detail |
|---|---|
| **Source** | Switch back to PowerPoint, advance to Slide 7 |
| **Action** | Single click to advance; hold on complete slide |
| **Narration** | "We are explicit about limitations…" (see script Slide 7) |
| **Notes** | Emphasis on "Label noise is *invisible*" — pause slightly |

---

## SHOT 11 — Slide 8: Conclusion  *(5:10 – 5:40)*
| Field | Detail |
|---|---|
| **Source** | PowerPoint full-screen, advance to Slide 8 |
| **Action** | Single click to advance; hold on complete slide; end on this slide |
| **Narration** | "In summary…" (see script Slide 8) |
| **Notes** | Hold 3 seconds of silence after "Thank you" before stopping recording |

---

## Post-recording
- [ ] Stop screen recorder
- [ ] Trim leading/trailing silence
- [ ] Export at 1920×1080 MP4 (H.264) or 720p if upload-limited
- [ ] Upload to project drive / attach to deliverable submission
- [ ] Screenshot of final slide → `docs/interface_screenshot.png` (if not already captured)
