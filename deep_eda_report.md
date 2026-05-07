# Deep Dive EDA: Why We "Discarded" 1.6 Billion Points

This report explains the reasoning behind our **22,000:1 Data Compression** strategy. By visualizing the raw sensor data, we can see why training a model on raw points is not only inefficient but mathematically noisy.

## 1. The Raw Signal (The "Wall of Noise")
**File**: `eda_images/raw_signal.png`

When you look at a single 10-second sample (320,000 points), it looks like a solid "wall" of vibration. 
*   **Problem**: At 32,000 samples per second, the physical motor cannot change its health state between point #100 and point #101. 
*   **Redundancy**: There is massive **autocorrelation**. 99.9% of these points are simply repeating the same oscillating pattern.

## 2. Zooming In (The Micro-Oscillation)
**File**: `eda_images/zoomed_signal.png`

Zooming into just 500 points (out of 320,000), we see a standard sine-wave-like vibration. 
*   **The "Waste"**: If we feed this raw wave into a machine learning model, the model spends all its "intelligence" trying to learn the shape of the sine wave (which we already know) rather than the **health of the motor**.
*   **Signal vs. Noise**: The exact position of the wave at microsecond #452 is "noise" (random timing). The **total amplitude** (height) of the wave is the "signal" (health).

## 3. Information Distillation (The Energy Feature)
**File**: `eda_images/energy_feature.png`

Instead of 320,000 random points, we calculate the **RMS (Root Mean Square)**. 
*   **What is it?**: It represents the "Energy" or "Total Vibration" of that 10-second window.
*   **Stability**: Notice how the energy feature (Visual 3) is a stable, smooth value. 
*   **Anomalies**: If the motor bearing starts to fail, this single "Energy" number will spike upwards. It is much easier for an algorithm to see a spike in **one number** than to find a pattern shift in **320,000 numbers**.

---

### **Conclusion: Distillation vs. Discarding**
We didn't "discard" the data in the sense of losing information. We **distilled** it. 

| Layer | Data Points | Information Type | Memory |
| :--- | :--- | :--- | :--- |
| **Raw Data** | 1.6 Billion | Oscillations / Noise | 13 GB |
| **Extracted Features** | 36,000 | Physical Health / Energy | 1 MB |

By using **14 features** (RMS, Skewness, Kurtosis, etc.), we captured the **soul** of the 13GB dataset. This allows our Isolation Forest to be:
1.  **Fast**: Inference takes milliseconds instead of seconds.
2.  **Accurate**: It's not distracted by the raw oscillating noise.
3.  **Lightweight**: The final model is small enough to run on any computer.

**You can view the generated plots in your project folder:**  
`d:\IOT Project\predictive-maintenance-ai\eda_images\`
