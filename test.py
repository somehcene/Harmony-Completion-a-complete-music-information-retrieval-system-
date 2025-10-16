import sys
import subprocess
from pathlib import Path
import argparse
import music21 as m21
from typing import List, Optional


# -------- Analyse des accords à partir d'un fichier MIDI --------
def extraire_accords_de_midi(chemin_fichier_midi: Path) -> Optional[List[str]]:
    """
    Lit un fichier MIDI et en extrait une liste de noms d'accords.

    Args:
        chemin_fichier_midi: chemin du .mid à analyser

    Returns:
        Liste de chaînes (ex: ['Cmajor', 'Aminor', ...]) ou None en cas d'erreur.
    """
    try:
        partition = m21.converter.parse(str(chemin_fichier_midi))
        partition_avec_accords = partition.chordify()

        liste_accords: List[str] = []
        for element in partition_avec_accords.recurse():
            if isinstance(element, m21.chord.Chord):
                try:
                    nom_accord = f"{element.root().name}{element.quality}"
                except Exception:
                    # fallback si jamais .root() / .quality pose souci
                    nom_accord = element.commonName or "UnknownChord"
                liste_accords.append(nom_accord)

        return liste_accords

    except m21.converter.ConverterException:
        print(f"Erreur : Impossible de lire le fichier MIDI à {chemin_fichier_midi}.")
    except Exception as e:
        print(f"Erreur inattendue lors de l'analyse MIDI : {e}")

    return None


# -------- Pipeline Basic Pitch -> MIDI -> Accords --------
def pipeline_transcription_basic_pitch(
    audio_input_path: Path,
    output_midi_dir: Path,
    overwrite: bool = False,
    onset_threshold: Optional[float] = None,
    frame_threshold: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    midi_tempo: Optional[int] = None,
) -> Optional[List[str]]:
    """Transcrit un fichier audio en MIDI avec Basic Pitch puis extrait des accords.

    Les paramètres optionnels permettent d'ajuster le comportement de Basic Pitch :
        - onset_threshold (0..1)
        - frame_threshold (0..1)
        - multiple_pitch_bends (bool)
        - midi_tempo (>0)
    """
    # 0) Vérifs basiques
    if not audio_input_path.exists():
        print(f"Erreur : Le fichier audio '{audio_input_path}' est introuvable.")
        return None

    output_midi_dir.mkdir(parents=True, exist_ok=True)

    # 1) Chemin MIDI attendu (<stem>_basic_pitch.mid)
    stem = audio_input_path.stem
    midi_path = output_midi_dir / f"{stem}_basic_pitch.mid"

    # 2) Gestion overwrite / réutilisation
    if midi_path.exists() and not overwrite:
        print(f"ℹ️  MIDI déjà présent, réutilisation : {midi_path}")
    else:
        if midi_path.exists() and overwrite:
            try:
                midi_path.unlink()
            except Exception as e:
                print(f"Attention : impossible de supprimer l'ancien MIDI ({midi_path}) : {e}")

        # ⚠️ ORDRE CORRECT : output_dir PUIS input_audio
        print(f"Transcription de {audio_input_path} en cours avec Basic Pitch...")
        commande = ["basic-pitch"]

        # --- Options CLI dynamiques ---
        if onset_threshold is not None:
            if not (0.0 <= onset_threshold <= 1.0):
                print("Erreur : --onset-threshold doit être entre 0 et 1.")
                return None
            commande += ["--onset-threshold", str(onset_threshold)]

        if frame_threshold is not None:
            if not (0.0 <= frame_threshold <= 1.0):
                print("Erreur : --frame-threshold doit être entre 0 et 1.")
                return None
            commande += ["--frame-threshold", str(frame_threshold)]

        if multiple_pitch_bends:
            commande += ["--multiple-pitch-bends"]

        if midi_tempo is not None:
            if midi_tempo <= 0:
                print("Erreur : --midi-tempo doit être un entier strictement positif.")
                return None
            commande += ["--midi-tempo", str(midi_tempo)]

        # Chemins (toujours en dernier)
        commande += [str(output_midi_dir), str(audio_input_path)]

        print("Commande exécutée :", " ".join(commande))

        try:
            subprocess.run(commande, check=True)
            print("Transcription terminée avec succès.")
        except FileNotFoundError:
            print("Erreur : la commande 'basic-pitch' est introuvable. "
                  "Installe-la dans cet environnement (ex : `pip install basic-pitch`).")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Une erreur est survenue lors de l'exécution de Basic Pitch : {e}")
            return None

        # 4) Fallback si le nom exact n’existe pas
        if not midi_path.exists():
            candidats = list(output_midi_dir.glob(f"{stem}_basic_pitch*.mid"))
            if candidats:
                candidats.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                midi_path = candidats[0]

    # 5) Vérifier l'existence finale du MIDI
    if not midi_path.exists():
        print(f"Erreur : Le fichier MIDI attendu est introuvable : {midi_path}")
        print("Assurez-vous que Basic Pitch a bien créé le .mid (voir logs).")
        return None

    print(f"Analyse du fichier MIDI généré ({midi_path}) pour les accords...")
    accords = extraire_accords_de_midi(midi_path)
    return accords


# -------- Point d'entrée --------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transcrire un audio en MIDI via Basic Pitch et extraire les accords.")
    p.add_argument("audio_file", type=Path, help="Chemin du fichier audio (wav/mp3/flac…)")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Dossier de sortie pour le MIDI")
    p.add_argument("--overwrite", action="store_true", help="Forcer la régénération du MIDI si déjà présent")

    p.add_argument("--onset-threshold", type=float, default=None,
                   help="(0..1) Probabilité minimale pour détecter un début de note")
    p.add_argument("--frame-threshold", type=float, default=None,
                   help="(0..1) Probabilité minimale pour soutenir une note")
    p.add_argument("--multiple-pitch-bends", action="store_true",
                   help="Autoriser des notes qui se chevauchent avec pitch bends (un instrument par hauteur)")
    p.add_argument("--midi-tempo", type=int, default=None, help="Tempo du fichier MIDI généré (BPM > 0)")

    return p


if __name__ == "__main__":
    # Exemple d'utilisation en CLI :
    #   python test.py recorded_audio3.wav --output-dir output --onset-threshold 0.5 --frame-threshold 0.3 \
    #          --multiple-pitch-bends --midi-tempo 96 --overwrite
    parser = _build_arg_parser()
    args = parser.parse_args()

    accords_trouves = pipeline_transcription_basic_pitch(
        audio_input_path=args.audio_file,
        output_midi_dir=args.output_dir,
        overwrite=args.overwrite,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        multiple_pitch_bends=args.multiple_pitch_bends,
        midi_tempo=args.midi_tempo,
    )

    if accords_trouves:
        print("\nAccords détectés :")
        print(accords_trouves)
    else:
        print("\nAucun accord n'a été détecté ou une erreur est survenue.")

