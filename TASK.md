## Project Tasks (Added: 2025-04-12)

- [x] **1. Hybrid Neural Network Architecture:** Implement a hybrid neural network combining collaborative and content-based filtering approaches for recommendations. *(Status: Appears implemented - requires review)*
- [x] **2. Data Utilization:** Integrate and utilize the synthetic user listening history data (~74k interactions) for training and evaluation. *(Status: Appears implemented - requires review)*
- [ ] **3. Explainable AI (XAI):** Implement XAI techniques (e.g., using Captum or SHAP) to understand model predictions.
- [ ] **4. Inference & Serving:** Develop an API endpoint (e.g., using FastAPI) for serving model predictions. *(Note: Basic Flask API exists for cold-start, consider migrating/integrating with FastAPI)*
- [~] **5. Cold Start Enhancement:** Research and implement improved strategies for handling the cold start problem (new users/items). *(Status: Basic implementation exists - requires review & enhancement)*
    - [ ] **5a. (New)** Enhance user profile creation (e.g., embedding mapping, better default handling, multi-preference aggregation).
    - [ ] **5b. (New)** Improve candidate generation (e.g., content similarity filtering instead of/addition to exact matching).
    - [ ] **5c. (New)** Refine recommendation strategy (e.g., hybrid scoring with popularity, add exploration/diversity mechanisms).
    - [ ] **5d. (New)** Consider alternative onboarding (e.g., item bootstrap rating).
    - [ ] **5e. (New)** Develop robust cold-start evaluation protocol & metrics.
- [ ] **6. Approximate Nearest Neighbors (ANN):** Investigate and potentially implement an ANN library (like Voyager or Faiss) for efficient similarity search, if applicable to the chosen model architecture. *(Note: Relevant for Task 5b)*
