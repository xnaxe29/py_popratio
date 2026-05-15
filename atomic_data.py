#import numpy as np
#from pprint import pprint


def read_blocks(filename):
    blocks = []
    current = []

    with open(filename, encoding="latin-1") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(line)

    if current:
        blocks.append(current)

    return blocks


def parse_levels(block):
    levels = []

    for line in block:
        cols = line.split()

        if len(cols) < 2:
            continue

        try:
            energy = float(cols[0])
            g = int(cols[1])
        except ValueError:
            continue

        levels.append({
            "energy": energy,
            "g": g,
        })

    return levels


def parse_A_transitions(block):
    transitions = []

    for line in block[1:]:
        cols = line.split()

        if len(cols) < 3:
            continue

        try:
            lower = int(cols[0])
            upper = int(cols[1])
            Aij = float(cols[2])
        except ValueError:
            continue

        transitions.append({
            "lower": lower,
            "upper": upper,
            "Aij": Aij,
        })

    return transitions


def parse_fluorescence(block):
    fluorescence = []

    for line in block[1:]:
        cols = line.split()

        if len(cols) < 4:
            continue

        try:
            energy = float(cols[0])
            g = int(cols[1])
            lower = int(cols[2])
            Aij = float(cols[3])
        except ValueError:
            continue

        fluorescence.append({
            "energy": energy,
            "g": g,
            "lower": lower,
            "Aij": Aij,
        })

    return fluorescence


def parse_collision_partner(block):
    name = block[0].strip()
    maxlevel = int(block[1])
    n_transitions_total = int(block[2])

    tables = []
    i = 3

    while i < len(block):
        cols = block[i].split()

        if len(cols) < 2:
            i += 1
            continue

        try:
            n_partial = int(cols[0])
            n_temp = int(cols[1])
        except ValueError:
            i += 1
            continue

        temperatures = [float(x) for x in block[i + 1].split()]
        i += 2

        for _ in range(n_partial):
            cols = block[i].split()

            level1 = int(cols[0])
            level2 = int(cols[1])
            rates = [float(x) for x in cols[2:]]

            if len(rates) != n_temp:
                raise ValueError(
                    f"{name}: expected {n_temp} rates, got {len(rates)} "
                    f"for transition {level1}->{level2}"
                )

            tables.append({
                "level1": level1,
                "level2": level2,
                "temperature": temperatures,
                "rate": rates,
            })

            i += 1

    if len(tables) != n_transitions_total:
        print(
            f"Warning: {name}: expected {n_transitions_total} transitions, "
            f"parsed {len(tables)}"
        )

    return {
        "name": name,
        "maxlevel": maxlevel,
        "n_transitions": n_transitions_total,
        "tables": tables,
    }


def parse_atom_file(filename):
    blocks = read_blocks(filename)

    #print("\nBlock structure:")
    #for i, block in enumerate(blocks):
    #    print(f"Block {i}: first line = {block[0]}")

    species_name = blocks[1][0].strip()
    n_ground = int(blocks[2][0])
    n_levels_max = int(blocks[3][0])

    levels = parse_levels(blocks[4])
    A_transitions = parse_A_transitions(blocks[5])
    fluorescence = parse_fluorescence(blocks[6])

    n_partners = int(blocks[7][0])
    partner_blocks = blocks[8:8 + n_partners]

    collision_partners = [
        parse_collision_partner(block)
        for block in partner_blocks
    ]

    atom = {
        "species_name": species_name,
        "n_ground": n_ground,
        "n_levels_max": n_levels_max,
        "levels": levels,
        "A": A_transitions,
        "fluorescence": fluorescence,
        "n_collision_partners": n_partners,
        "collision_partners": collision_partners,
    }

    return atom



if __name__ == "__main__":
    print("atomic data module loaded successfully")

'''
if __name__ == "__main__":
    atom = parse_atom_file("OI.dat")

    print("\nParsed basic atom information:")
    print("species_name =", atom["species_name"])
    print("n_ground =", atom["n_ground"])
    print("n_levels_max =", atom["n_levels_max"])
    print("number of levels parsed =", len(atom["levels"]))
    print("number of A transitions parsed =", len(atom["A"]))
    print("number of fluorescence transitions parsed =", len(atom["fluorescence"]))
    print("number of collision partners =", atom["n_collision_partners"])

    print("\nFirst few levels:")
    pprint(atom["levels"][:5])

    print("\nFirst few A transitions:")
    pprint(atom["A"][:5])

    print("\nFirst few fluorescence transitions:")
    pprint(atom["fluorescence"][:5])

    print("\nCollision partners:")
    for partner in atom["collision_partners"]:
        print(
            partner["name"],
            "maxlevel =",
            partner["maxlevel"],
            "tables =",
            len(partner["tables"])
        )

    print("\nFirst table of first collision partner:")
    pprint(atom["collision_partners"][0]["tables"][0])

'''
