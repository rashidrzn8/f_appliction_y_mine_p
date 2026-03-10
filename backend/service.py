import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ClassificationService:
    """
    Multi-domain text classification service.

    Each 'domain' maps to a separate set of:
        model.h5  /  model_<domain>.h5
        tokenizer.pkl  /  tokenizer_<domain>.pkl
        label_encoder.pkl  /  label_encoder_<domain>.pkl

    The 'base' domain uses the un-suffixed filenames for
    backwards compatibility with your existing saved models.
    """

    # Sequence length used at training time for the BiLSTM models.
    # Change here if you retrain with a different max_sequence_length.
    BILSTM_MAX_LEN  = 20    # Sinhala BiLSTM models (normal / augmented)
    DEFAULT_MAX_LEN = 100   # auto-trained models via train_model()

    def __init__(self):
        # domain_name → { 'model', 'tokenizer', 'label_encoder', 'max_len' }
        self.domains: dict = {}
        self.errors:  list = []

        # Load the default base domain on startup
        self.load_resources("base")

    # ──────────────────────────────────────────────────────────
    #  RESOURCE LOADING
    # ──────────────────────────────────────────────────────────
    def load_resources(self, domain: str = "base") -> bool:
        """
        Load model + tokenizer + label_encoder for the given domain.

        File naming convention:
            domain == 'base'          →  model.h5, tokenizer.pkl, label_encoder.pkl
            domain == 'normal'        →  model_normal.h5, tokenizer_normal.pkl, ...
            domain == 'augmented'     →  model_augmented.h5, ...
            domain == 'base_augmented'→  model_base_augmented.h5, ...

        Returns True on success, False on failure.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"\n[ClassificationService] Loading domain '{domain}' from: {current_dir}")

        prefix = "" if domain == "base" else f"_{domain}"

        paths = {
            'model':         os.path.join(current_dir, f'model{prefix}.h5'),
            'tokenizer':     os.path.join(current_dir, f'tokenizer{prefix}.pkl'),
            'label_encoder': os.path.join(current_dir, f'label_encoder{prefix}.pkl'),
        }

        resources = {'model': None, 'tokenizer': None, 'label_encoder': None}
        success = True

        # ── Model ────────────────────────────────────────────
        if os.path.exists(paths['model']):
            try:
                print(f"  Loading model      : {paths['model']}")
                resources['model'] = tf.keras.models.load_model(paths['model'])
            except Exception as e:
                msg = f"[{domain}] Failed to load model: {e}"
                self.errors.append(msg)
                print(f"  ERROR: {msg}")
                success = False
        else:
            msg = f"[{domain}] Model file not found: {paths['model']}"
            self.errors.append(msg)
            print(f"  WARNING: {msg}")
            success = False

        # ── Tokenizer ────────────────────────────────────────
        if os.path.exists(paths['tokenizer']):
            try:
                print(f"  Loading tokenizer  : {paths['tokenizer']}")
                with open(paths['tokenizer'], 'rb') as f:
                    resources['tokenizer'] = pickle.load(f)
            except Exception as e:
                msg = f"[{domain}] Failed to load tokenizer: {e}"
                self.errors.append(msg)
                print(f"  ERROR: {msg}")
                success = False
        else:
            msg = f"[{domain}] Tokenizer file not found: {paths['tokenizer']}"
            self.errors.append(msg)
            print(f"  WARNING: {msg}")
            success = False

        # ── Label encoder ────────────────────────────────────
        if os.path.exists(paths['label_encoder']):
            try:
                print(f"  Loading encoder    : {paths['label_encoder']}")
                with open(paths['label_encoder'], 'rb') as f:
                    resources['label_encoder'] = pickle.load(f)
            except Exception as e:
                msg = f"[{domain}] Failed to load label encoder: {e}"
                self.errors.append(msg)
                print(f"  ERROR: {msg}")
                success = False
        else:
            msg = f"[{domain}] Label encoder file not found: {paths['label_encoder']}"
            self.errors.append(msg)
            print(f"  WARNING: {msg}")
            success = False

        # ── Store only if all three loaded ───────────────────
        if success and all(v is not None for v in resources.values()):
            # Choose correct max_len based on domain name
            if any(tag in domain for tag in ('normal', 'augmented', 'sinhala', 'bilstm')):
                resources['max_len'] = self.BILSTM_MAX_LEN
            else:
                resources['max_len'] = self.DEFAULT_MAX_LEN

            self.domains[domain] = resources
            print(f"  ✅ Domain '{domain}' ready (max_len={resources['max_len']}).")
            return True
        else:
            print(f"  ❌ Domain '{domain}' could not be fully loaded.")
            return False

    # ──────────────────────────────────────────────────────────
    #  PREDICTION
    # ──────────────────────────────────────────────────────────
    def predict(self, text: str, domain: str = "base") -> dict:
        """
        Classify text using the given domain model.

        Returns:
            {
                "class":         str,
                "confidence":    float,
                "probabilities": [ {"class": str, "probability": float}, ... ]
            }
        or  {"error": str, "details": ...}  on failure.
        """
        # Lazy-load domain if not yet in memory
        if domain not in self.domains:
            loaded = self.load_resources(domain)
            if not loaded:
                return {
                    "error": f"Model for domain '{domain}' could not be loaded.",
                    "details": self.errors
                }

        resources = self.domains[domain]
        max_len   = resources.get('max_len', self.DEFAULT_MAX_LEN)

        try:
            # ── Preprocess ──────────────────────────────────
            sequence        = resources['tokenizer'].texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

            # ── Predict ─────────────────────────────────────
            prediction          = resources['model'].predict(padded_sequence, verbose=0)[0]
            predicted_idx       = int(np.argmax(prediction))

            encoder = resources['label_encoder']

            # ── Build probabilities list ─────────────────────
            all_probabilities = []
            for i, prob in enumerate(prediction):
                if hasattr(encoder, 'inverse_transform'):
                    cls_name = str(encoder.inverse_transform([i])[0])
                else:
                    cls_name = str(i)
                all_probabilities.append({
                    "class":       cls_name,
                    "probability": float(prob)
                })

            all_probabilities.sort(key=lambda x: x['probability'], reverse=True)

            # ── Decode predicted label ───────────────────────
            if hasattr(encoder, 'inverse_transform'):
                class_label = str(encoder.inverse_transform([predicted_idx])[0])
            else:
                class_label = str(predicted_idx)

            return {
                "class":         class_label,
                "confidence":    float(np.max(prediction)),
                "probabilities": all_probabilities
            }

        except Exception as e:
            print(f"[ClassificationService] Prediction error for domain '{domain}': {e}")
            return {"error": "Prediction failed", "details": str(e)}

    # ──────────────────────────────────────────────────────────
    #  ON-DEMAND TRAINING
    # ──────────────────────────────────────────────────────────
    def train_model(self, csv_path: str, domain: str) -> dict:
        """
        Train a lightweight classification model from a CSV file and
        save + load it under the given domain name.

        CSV must contain a text column and a label column.
        Column names are auto-detected using common naming conventions.
        """
        try:
            import pandas as pd
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.utils.class_weight import compute_class_weight
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping

            print(f"\n[ClassificationService] Training domain '{domain}' with: {csv_path}")
            df = pd.read_csv(csv_path)
            df.columns = df.columns.astype(str).str.lower().str.strip()

            # ── Auto-detect text column ──────────────────────
            text_col_candidates  = ['text', 'title', 'tweet', 'sentence', 'document',
                                     'body', 'content', 'review', 'message']
            label_col_candidates = ['label', 'class', 'category', 'tag',
                                     'target', 'sentiment', 'topic']

            text_col  = next((c for c in text_col_candidates  if c in df.columns), None)
            label_col = next((c for c in label_col_candidates if c in df.columns), None)

            # Fallback heuristic
            if not text_col or not label_col:
                print("  Falling back to heuristic column detection...")
                string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                if len(string_cols) < 2:
                    string_cols = df.columns.tolist()

                if len(string_cols) >= 2:
                    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in string_cols}
                    text_col    = max(avg_lengths, key=avg_lengths.__getitem__)
                    string_cols.remove(text_col)
                    unique_counts = {col: df[col].nunique() for col in string_cols}
                    label_col     = min(unique_counts, key=unique_counts.__getitem__)

            if not text_col or not label_col:
                return {
                    "error": (
                        f"Could not auto-detect text/label columns. "
                        f"Found: {list(df.columns)}. "
                        f"Please rename your columns to 'text' and 'label'."
                    )
                }

            print(f"  Text column: '{text_col}' | Label column: '{label_col}'")
            texts  = df[text_col].astype(str).tolist()
            labels = df[label_col].astype(str).tolist()

            # ── Encode labels ────────────────────────────────
            encoder = LabelEncoder()
            y       = encoder.fit_transform(labels)
            num_classes = len(encoder.classes_)
            print(f"  Classes ({num_classes}): {list(encoder.classes_)}")

            # ── Tokenise ─────────────────────────────────────
            max_words = 10000
            max_len   = self.DEFAULT_MAX_LEN
            tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            X = pad_sequences(sequences, maxlen=max_len, padding='post')

            # ── Train / val split ────────────────────────────
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )

            # ── Class weights ────────────────────────────────
            cw_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            cw_dict  = dict(enumerate(cw_array))

            # ── Build model ──────────────────────────────────
            model = Sequential([
                Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
                GlobalAveragePooling1D(),
                Dense(64, activation='relu'),
                Dropout(0.4),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # ── Train ────────────────────────────────────────
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model.fit(
                X_train, y_train,
                epochs=15,
                batch_size=32,
                validation_data=(X_val, y_val),
                class_weight=cw_dict,
                callbacks=[early_stop],
                verbose=1
            )

            # ── Save artefacts ───────────────────────────────
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prefix = "" if domain == "base" else f"_{domain}"

            model_path   = os.path.join(current_dir, f'model{prefix}.h5')
            tok_path     = os.path.join(current_dir, f'tokenizer{prefix}.pkl')
            enc_path     = os.path.join(current_dir, f'label_encoder{prefix}.pkl')

            model.save(model_path)
            with open(tok_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            with open(enc_path, 'wb') as f:
                pickle.dump(encoder, f)

            print(f"  Saved: {model_path}")

            # ── Load into memory ─────────────────────────────
            self.load_resources(domain)

            return {
                "message": f"Model trained and loaded for domain '{domain}'",
                "domain":  domain,
                "classes": list(encoder.classes_)
            }

        except Exception as e:
            print(f"[ClassificationService] Training error: {e}")
            return {"error": "Training failed", "details": str(e)}


# ── Singleton ──────────────────────────────────────────────────
classification_service = ClassificationService()