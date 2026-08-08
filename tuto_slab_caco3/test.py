"""
Script pour générer la surface (10-14) de la calcite à partir d'un fichier CIF.
"""

from pymatgen.core.surface import SlabGenerator
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def generate_calcite_surface(
    cif_path: str,
    miller_indices=(1, 0, 4),
    min_slab_size: float = 25.0,
    min_vacuum_size: float = 15.0,
    max_broken_bonds: int = 10,
    symmetrize: bool = True,
    output_file: str = "calcite_10-14_slab.cif",
):
    """
    Génère une surface de calcite à partir d'un fichier CIF.

    Parameters
    ----------
    cif_path : str
        Chemin vers le fichier CIF de la calcite.
    miller_indices : tuple
        Indices de Miller pour la surface (par défaut: (1, 0, -1, 4) pour (10-14)).
    min_slab_size : float
        Épaisseur minimale du slab en Å.
    min_vacuum_size : float
        Épaisseur minimale du vide en Å.
    max_broken_bonds : int
        Nombre maximum de liaisons cassées autorisées.
    symmetrize : bool
        Si True, symétrise le slab.
    output_file : str
        Nom du fichier de sortie.

    Returns
    -------
    tuple
        (slab, slabs_list, shifts, dipoles, broken_bonds)
    """
    print("=" * 60)
    print("Génération de la surface (10-14) de la calcite")
    print("=" * 60)


"""
Script corrigé pour générer la surface (10-14) de la calcite.
"""


def generate_calcite_surface(
    cif_path: str,
    miller_indices=(1, 0, 4),
    min_slab_size: float = 20.0,  # Réduit
    min_vacuum_size: float = 15.0,
    max_broken_bonds: int = 20,  # Augmenté
    symmetrize: bool = False,  # Désactivé pour commencer
    output_file: str = "calcite_10-14_slab.cif",
):
    """
    Génère une surface de calcite avec des paramètres optimisés.
    """
    print("=" * 60)
    print("Génération de la surface (10-14) de la calcite")
    print("=" * 60)

    # 1. Charger la structure
    print(f"\n1. Chargement du fichier CIF: {cif_path}")
    parser = CifParser(cif_path)
    structure = parser.parse_structures()[0]

    # Raffiner la structure
    print("   Raffinement de la structure...")
    analyzer = SpacegroupAnalyzer(structure)
    structure = analyzer.get_refined_structure()

    print(f"   Structure: {structure.formula}")
    print(f"   Groupe d'espace: {structure.get_space_group_info()[0]}")
    print(f"   Nombre d'atomes: {len(structure)}")
    print(f"   Paramètres: {structure.lattice.abc} Å")

    # 2. Ajouter les états d'oxydation
    try:
        structure.add_oxidation_state_by_guess()
        print("\n2. États d'oxydation ajoutés.")
    except Exception as e:
        print(f"\n2. Impossible d'ajouter les états d'oxydation: {e}")

    # 3. Définir les liaisons - plus complètes
    bonds = {
        # Avec états d'oxydation
        ("Ca2+", "O2-"): 2.5,
        ("C4+", "O2-"): 1.4,
        # Sans états d'oxydation
        ("Ca", "O"): 2.5,
        ("C", "O"): 1.4,
        ("O", "Ca"): 2.5,
        ("O", "C"): 1.4,
    }

    print(f"\n3. Génération du slab avec les indices {miller_indices}")
    print(f"   Épaisseur min: {min_slab_size} Å")
    print(f"   Vide min: {min_vacuum_size} Å")
    print(f"   Liaisons cassées max: {max_broken_bonds}")

    # 4. Créer le générateur avec plus de flexibilité
    slabgen = SlabGenerator(
        structure,
        miller_indices,
        min_slab_size,
        min_vacuum_size,
        primitive=True,
        lll_reduce=True,
    )

    # 5. Essayer différentes approches
    all_slabs = []

    # Approche 2: Sans bonds
    if not all_slabs:
        print("\n5. Tentative sans liaisons...")
        try:
            slabs = slabgen.get_slabs()
            if slabs:
                print(f"   ✓ Trouvé {len(slabs)} slabs")
                all_slabs.extend(slabs)
        except Exception as e:
            print(f"   ✗ Échec: {e}")

    # Approche 3: Avec une plus petite épaisseur
    if not all_slabs:
        print("\n6. Tentative avec épaisseur réduite (15 Å)...")
        slabgen_reduced = SlabGenerator(
            structure,
            miller_indices,
            15.0,  # Plus petite épaisseur
            min_vacuum_size,
            primitive=True,
            lll_reduce=True,
        )
        try:
            slabs = slabgen_reduced.get_slabs()
            if slabs:
                print(f"   ✓ Trouvé {len(slabs)} slabs")
                all_slabs.extend(slabs)
        except Exception as e:
            print(f"   ✗ Échec: {e}")

    # Approche 4: Sans primitive
    if not all_slabs:
        print("\n7. Tentative sans primitive (conventionnelle)...")
        slabgen_conv = SlabGenerator(
            structure,
            miller_indices,
            min_slab_size,
            min_vacuum_size,
            primitive=False,
            lll_reduce=True,
        )
        try:
            slabs = slabgen_conv.get_slabs()
            if slabs:
                print(f"   ✓ Trouvé {len(slabs)} slabs")
                all_slabs.extend(slabs)
        except Exception as e:
            print(f"   ✗ Échec: {e}")

    if not all_slabs:
        raise RuntimeError(
            "Aucun slab n'a pu être généré.\n"
            "Suggestions:\n"
            "  1. Vérifiez que le fichier CIF est valide\n"
            "  2. Essayez une autre surface (ex: (0, 0, 1))\n"
            "  3. Utilisez une version plus récente de pymatgen"
        )

    # 6. Sélectionner le meilleur slab
    best_slab = all_slabs[0]
    best_dipole = abs(best_slab.dipole[2]) if hasattr(best_slab, "dipole") else 0

    for slab in all_slabs[1:]:
        dipole = abs(slab.dipole[2]) if hasattr(slab, "dipole") else 0
        if dipole < best_dipole:
            best_dipole = dipole
            best_slab = slab

    print("\n8. Résultat:")
    print(f"   Meilleur slab: {len(best_slab)} atomes")
    print(f"   Dipole: {best_dipole:.4f} D")
    print(f"   Maille: {best_slab.lattice.abc} Å")
    print(f"   Angles: {best_slab.lattice.angles} °")

    # 7. Sauvegarder
    best_slab.to_file(output_file)
    print(f"\n9. Sauvegardé dans: {output_file}")

    return best_slab


def save_lammps_data(slab, output_file="calcite_10-14_slab.lmp"):
    """Sauvegarde le slab au format LAMMPS data."""
    try:
        from pymatgen.io.lammps.data import LammpsData

        print(f"\nSauvegarde au format LAMMPS: {output_file}")
        lmp_data = LammpsData.from_structure(slab)
        lmp_data.write_file(output_file)
    except ImportError:
        print("Impossible de sauvegarder au format LAMMPS.")
        print("Assurez-vous que pymatgen est à jour.")

    # Afficher un résumé des propriétés
    print("\nRésumé des propriétés du slab:")
    print(f"  Formule: {slab.formula}")
    print(f"  Nombre d'atomes: {len(slab)}")
    print(f"  Volume: {slab.volume:.2f} Å³")
    print(f"  Densité: {slab.density:.2f} g/cm³")


generate_calcite_surface("9016705.cif")
