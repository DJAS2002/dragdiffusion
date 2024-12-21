# Installatie van drag diffusion

- Python 3.10 was reeds aanwezig op de VM. Vandaaruit gaan werken.

- Alle requirements van `environment.yaml` naar `requirements.txt` gekopieerd.

- Alle torch gerelateerde package installs weggehaald uit requirements.txt

- Handmatig de juiste torch met CUDA geïnstalleerd. Omdat de laatste versie gebruikt wordt was het niet nodig om de versie aan te geven: ```pip3 install torch torchvision torchaudio```

- Diverse versies gewijzigd, n.a.v. fouten tijdens het runnen van scripts. Aanpassingen staan in requirements.txt

## notes
- De GUI gemaakt met `gradio` werkte niet met de voorgestelde versie. Geupgrade naar versie 3.50.2, waardoor ook `huggingface-hub<0.26.0` moest worden en `jinja2==3.1.2`