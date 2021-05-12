# Estimation du déclin des espèces à partir de données génomiques

## Prérequis

L'utilisation de Miniconda3 est fortement recommandée pour l'utilisation de ce programme.

## Quick start

1. Clone du répertoire github

> Lien HTTPS

```
https://github.com/Pierre-damase/Species_evolution_inference.git
```

2. Initialiser l'environnement conda à partir du fichier sei-3.8.5.yml

```
conda env create --file sei-3.8.5.yml
```

3. Activer l'environnement conda

```
conda activate sei-3.8.5
```

4. Installation de SMC++

Librairie nécessaire.

```
sudo apt install -y python3-dev libgmp-dev libmpfr-dev libgsl0-dev
```

Installation préalable de Cython nécessaire.

```
pip install Cython
```

Installation de SMC++

```
pip install git+https://github.com/popgenmethods/smcpp
```

5. Utilisation

## Auteur

IMBERT Pierre
