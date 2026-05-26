import streamlit as st
import joblib
import numpy as np
import pandas as pd
from collections import Counter
import lime
import lime.lime_tabular

# Config page
st.set_page_config(
    page_title="AI4FA ",
    page_icon="",
    layout="wide"
)

# Importer modèle
@st.cache_resource
def load_model():
    data = joblib.load("random_forest(3).pkl")
    return data["model"], data["seuil"], data["ranking"]

model, seuil, ranking = load_model()

# Données synthétiques pour LIME (puisque X_train n'est pas dans le pkl)
# On génère un jeu de données représentatif basé sur les plages médicales connues
@st.cache_resource
def build_lime_explainer(_model):
    np.random.seed(42)
    n = 500

    # Variables binaires (0 ou 1)
    binary_cols = [
        'Palpitations', 'Sexe', 'Tabac', 'HTA', 'Hypercholesterolémie',
        'DNID', 'OH', 'Flutter', 'Rao', 'RM', 'IM', 'BPCO',
        'Cardiopathie ischémique', 'IDM', 'SAS', 'artérite', 'IC',
        'Hyperthiroidie', 'HVG/CMH', 'Anévrisme Ao', 'ESA', 'AVC/AIT',
        'Athérome carotidien', 'Atcd fam - DNID', 'Atcd fam - HTA',
        'Atcd fam - AVC', 'Atcd fam - troubles rythme ', 'Atcd fam - FA',
        'Atcd fam - IDM'
    ]

    # Variables continues avec plages médicales réalistes
    continuous_ranges = {
        'Age':          (30, 90),
        'Poids(kg)':    (45, 130),
        'Taille(cm)':   (150, 195),
        'IMC':          (16, 45),
        'FC (batt/min)':(40, 120),
        'P (ms)':       (60, 160),
        'PR (ms)':      (100, 280),
        'QRS (ms)':     (60, 160),
        'QT (ms)':      (300, 550),
        'QTc (ms)':     (350, 550),
        'QRS (°)':      (-90, 180),
    }

    data_synth = {}

    for col in FEATURE_NAMES:
        if col in binary_cols:
            data_synth[col] = np.random.randint(0, 2, n).astype(float)
        elif col in continuous_ranges:
            lo, hi = continuous_ranges[col]
            data_synth[col] = np.random.uniform(lo, hi, n)
        else:
            data_synth[col] = np.random.uniform(0, 1, n)

    X_synth = pd.DataFrame(data_synth, columns=FEATURE_NAMES)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_synth.values,
        feature_names=FEATURE_NAMES,
        class_names=['Sain', 'Malade'],
        mode='classification',
        discretize_continuous=True
    )
    return explainer

# Explicabilité consensus
def summarizer_consensus(model, X_sample, feature_names, threshold):
    prob_positive = model.predict_proba(X_sample)[0][1]
    final_pred = 1 if prob_positive >= threshold else 0

    X_values = X_sample.values
    individual_preds = [tree.predict(X_values)[0] for tree in model.estimators_]
    winning_trees_idx = [i for i, p in enumerate(individual_preds) if p == final_pred]

    all_rules = []
    for idx in winning_trees_idx:
        tree = model.estimators_[idx]
        node_indicator = tree.decision_path(X_values)
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        for node_id in node_index:
            if tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]:
                feature_idx = tree.tree_.feature[node_id]
                feature_name = feature_names[feature_idx]
                th_tree = tree.tree_.threshold[node_id]
                val = X_values[0][feature_idx]

                if val <= th_tree:
                    all_rules.append(f"{feature_name} ≤ {th_tree:.2f}")
                else:
                    all_rules.append(f"{feature_name} > {th_tree:.2f}")

    rules_counter = Counter(all_rules).most_common()

    return {
        "prob_positive": prob_positive,
        "final_pred": final_pred,
        "n_winning": len(winning_trees_idx),
        "n_total": len(model.estimators_),
        "rules": rules_counter,
    }

# Interface
st.title("AI4FA")
st.markdown("Renseignez les informations du patient, puis cliquez sur le bouton Prédire.")
st.divider()


# SECTION 1
st.subheader("Données biométriques et ECG")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    p    = st.number_input("P (ms)",        min_value=0.0, value=None)
    pr   = st.number_input("PR (ms)",       min_value=0.0, value=None)
with c2:
    qtc  = st.number_input("QTc (ms)",      min_value=0.0, value=None)
    qt   = st.number_input("QT (ms)",       min_value=0.0, value=None)
with c3:
    qrs_ms  = st.number_input("QRS (ms)",   min_value=0.0, value=None)
    qrs_deg = st.number_input("QRS (°)",    min_value=-180.0, max_value=180.0, value=None)
with c4:
    age  = st.number_input("Âge (ans)",     min_value=0, max_value=120, value=None, step=1)
    imc  = st.number_input("IMC",           min_value=0.0, value=None)
with c5:
    poids = st.number_input("Poids (kg)",   min_value=0.0, value=None)
    taille= st.number_input("Taille (cm)",  min_value=0.0, value=None)

c6, c7, c8, c9, c10 = st.columns(5)
with c6:
    fc   = st.number_input("FC (batt/min)", min_value=0.0, value=None)
with c7:
    sexe = st.selectbox("Sexe", options=[None, "Homme", "Femme"],
                        format_func=lambda x: "Sélectionner..." if x is None else x)

st.divider()


# SECTION 2
st.subheader("Antécédents et facteurs de risque")

def oui_non(label):
    return st.selectbox(label, options=[None, "Non", "Oui"],
                        format_func=lambda x: "—" if x is None else x,
                        key=label)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Antécédents médicaux/cardiovasculaires**")
    ic               = oui_non("IC")
    avc_ait          = oui_non("AVC/AIT")
    atherome         = oui_non("Athérome carotidien")
    cardiopathie     = oui_non("Cardiopathie ischémique")
    anevrisme        = oui_non("Anévrisme Ao")
    arterite         = oui_non("artérite")
    rm               = oui_non("RM")
    im               = oui_non("IM")
    esa              = oui_non("ESA")
    hvg_cmh          = oui_non("HVG/CMH")
    idm              = oui_non("IDM")
    bpco             = oui_non("BPCO")
    sas              = oui_non("SAS")
    rao              = oui_non("Rao")
    hyperthyroidie   = oui_non("Hyperthiroidie")
    flutter          = oui_non("Flutter")
    palpitations     = oui_non("Palpitations")

with col2:
    st.markdown("**Antécédents familiaux**")
    atcd_fa          = oui_non("Atcd fam - FA")
    atcd_avc         = oui_non("Atcd fam - AVC")
    atcd_hta         = oui_non("Atcd fam - HTA")
    atcd_idm         = oui_non("Atcd fam - IDM")
    atcd_dnid        = oui_non("Atcd fam - DNID")
    atcd_rythme      = oui_non("Atcd fam - troubles rythme")

with col3:
    st.markdown("**Facteurs de risque**")
    hypercholesterol = oui_non("Hypercholestérolémie")
    tabac            = oui_non("Tabac")
    dnid             = oui_non("DNID")
    oh               = oui_non("OH")
    hta              = oui_non("HTA")

st.divider()


# Prédiction
def encode_bin(val):
    if val is None:
        return None
    if val == "Homme":
        return 0
    if val == "Femme":
        return 1
    return 1 if val == "Oui" else 0


FEATURE_NAMES = [
    'Palpitations',
    'Age',
    'Sexe',
    'Poids(kg)',
    'Taille(cm)',
    'IMC',
    'Tabac',
    'HTA',
    'Hypercholesterolémie',
    'DNID',
    'OH',
    'Flutter',
    'Rao',
    'RM',
    'IM',
    'BPCO',
    'Cardiopathie ischémique',
    'IDM',
    'SAS',
    'artérite',
    'IC',
    'Hyperthiroidie',
    'HVG/CMH',
    'Anévrisme Ao',
    'ESA',
    'AVC/AIT',
    'Athérome carotidien',
    'Atcd fam - DNID',
    'Atcd fam - HTA',
    'Atcd fam - AVC',
    'Atcd fam - troubles rythme ',
    'Atcd fam - FA',
    'Atcd fam - IDM',
    'FC (batt/min)',
    'P (ms)',
    'PR (ms)',
    'QRS (ms)',
    'QT (ms)',
    'QTc (ms)',
    'QRS (°)',
]

# Dictionnaire valeur brute pour chaque feature
valeurs_dict = {
    'Palpitations':               encode_bin(palpitations),
    'Age':                        age,
    'Sexe':                       encode_bin(sexe),
    'Poids(kg)':                  poids,
    'Taille(cm)':                 taille,
    'IMC':                        imc,
    'Tabac':                      encode_bin(tabac),
    'HTA':                        encode_bin(hta),
    'Hypercholesterolémie':       encode_bin(hypercholesterol),
    'DNID':                       encode_bin(dnid),
    'OH':                         encode_bin(oh),
    'Flutter':                    encode_bin(flutter),
    'Rao':                        encode_bin(rao),
    'RM':                         encode_bin(rm),
    'IM':                         encode_bin(im),
    'BPCO':                       encode_bin(bpco),
    'Cardiopathie ischémique':    encode_bin(cardiopathie),
    'IDM':                        encode_bin(idm),
    'SAS':                        encode_bin(sas),
    'artérite':                   encode_bin(arterite),
    'IC':                         encode_bin(ic),
    'Hyperthiroidie':             encode_bin(hyperthyroidie),
    'HVG/CMH':                    encode_bin(hvg_cmh),
    'Anévrisme Ao':               encode_bin(anevrisme),
    'ESA':                        encode_bin(esa),
    'AVC/AIT':                    encode_bin(avc_ait),
    'Athérome carotidien':        encode_bin(atherome),
    'Atcd fam - DNID':            encode_bin(atcd_dnid),
    'Atcd fam - HTA':             encode_bin(atcd_hta),
    'Atcd fam - AVC':             encode_bin(atcd_avc),
    'Atcd fam - troubles rythme ': encode_bin(atcd_rythme),
    'Atcd fam - FA':              encode_bin(atcd_fa),
    'Atcd fam - IDM':             encode_bin(atcd_idm),
    'FC (batt/min)':              fc,
    'P (ms)':                     p,
    'PR (ms)':                    pr,
    'QRS (ms)':                   qrs_ms,
    'QT (ms)':                    qt,
    'QTc (ms)':                   qtc,
    'QRS (°)':                    qrs_deg,
}

champs_vides = any(v is None for v in valeurs_dict.values())

if st.button("Prédire", type="primary", use_container_width=True, disabled=champs_vides):
    # Prédiction
    X = pd.DataFrame([valeurs_dict], columns=FEATURE_NAMES).astype(float)

    probas = model.predict_proba(X)[0]
    label  = 1 if probas[1] >= seuil else 0
    confiance = probas.max()

    st.subheader("Résultat")
    st.caption(f"Seuil de décision appliqué : **{seuil:.2f}**   Probabilité que le patient soit malade : **{probas[1]:.1%}**")

    if label == 1:
        st.error(f"**Diagnostic suggéré : Malade** avec une confiance de {confiance:.0%}")
    else:
        st.success(f"**Diagnostic suggéré : Sain** avec une confiance de {confiance:.0%}")

    classes = model.classes_
    with st.expander("Détail des probabilités"):
        for cls, proba in zip(classes, probas):
            label_display = "Malade" if cls == 1 else "Sain"
            st.write(f"**{label_display}** : {proba:.1%}")
            st.progress(float(proba))

    st.divider()

    # Explicabilité
    st.subheader("Explication de la décision")

    consensus = summarizer_consensus(model, X, FEATURE_NAMES, threshold=seuil)

    st.markdown(
        f"**{consensus['n_winning']} arbres sur {consensus['n_total']}** ont voté pour la classe "
        f"**{'Malade' if consensus['final_pred'] == 1 else 'Sain'}**."
    )

    tab1, tab2, tab3 = st.tabs([
        "Par fréquence d'apparition",
        "Par importance de feature",
        "Inteprétation LIME"
    ])

    if not consensus["rules"]:
        st.info("Aucune règle intermédiaire identifiée (prédiction directement à la racine).")
    else:
        # Onglet 1 : classement par fréquence d'apparition
        with tab1:
            st.markdown("Règles classées par nombre d'arbres décisifs les ayant utilisées.")
            max_count = consensus["rules"][0][1]
            for rule, count in consensus["rules"]:
                pct = count / consensus["n_winning"]
                st.markdown(f"`{rule}` vue dans **{count}** arbre(s) ({pct:.0%})")
                st.progress(count / max_count)

        # Onglet 2 : classement par importance de feature
        with tab2:
            st.markdown("Règles classées par importance globale de la feature lors de l'entraînement.")

            def get_feature_importance(rule):
                for feature_name in ranking:
                    if rule.startswith(feature_name):
                        return ranking[feature_name]
                return 0.0

            rules_by_importance = sorted(
                consensus["rules"],
                key=lambda x: get_feature_importance(x[0]),
                reverse=True
            )
            max_importance = max(get_feature_importance(r) for r, _ in rules_by_importance)
            for rule, count in rules_by_importance:
                importance = get_feature_importance(rule)
                pct = count / consensus["n_winning"]
                st.markdown(f"`{rule}` importance feature : **{importance:.4f}** · vue dans **{count}** arbre(s) ({pct:.0%})")
                st.progress(importance / max_importance if max_importance > 0 else 0)

    # Onglet 3 : LIME — EN DEHORS du else, toujours affiché
    with tab3:
        st.markdown(
            "**LIME** (Local Interpretable Model-agnostic Explanations) explique "
            "la contribution de chaque variable à la prédiction pour ce patient précis."
        )
        st.caption("🔴 = pousse vers Malade · 🟢 = pousse vers Sain")

        with st.spinner("Calcul de l'explication LIME en cours..."):
            lime_explainer = build_lime_explainer(model)
            exp = lime_explainer.explain_instance(
                X.values[0],
                model.predict_proba,
                num_features=15
            )

        lime_list = exp.as_list()
        lime_df = pd.DataFrame(lime_list, columns=["Règle", "Contribution"])
        lime_df = lime_df.reindex(
            lime_df["Contribution"].abs().sort_values(ascending=False).index
        )

        max_contrib = lime_df["Contribution"].abs().max()
        for _, row in lime_df.iterrows():
            signe = "🔴" if row["Contribution"] > 0 else "🟢"
            val = row["Contribution"]
            barre = abs(val) / max_contrib if max_contrib > 0 else 0
            st.markdown(f"{signe} `{row['Règle']}` → **{val:+.4f}**")
            st.progress(barre)

        with st.expander("Voir le tableau complet des contributions LIME"):
            lime_df_display = lime_df.copy()
            lime_df_display["Direction"] = lime_df_display["Contribution"].apply(
                lambda v: "→ Malade" if v > 0 else "→ Sain"
            )
            lime_df_display["Contribution"] = lime_df_display["Contribution"].map("{:+.4f}".format)
            st.dataframe(lime_df_display, use_container_width=True, hide_index=True)