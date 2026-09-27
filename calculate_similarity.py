import os
import torch
import torch.nn.functional as F


# ============================================================
# 1. Path
# ============================================================

ACTIVATION_DIR = "activation"
OUTPUT_DIR = "similarity"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. Dataset names
# ============================================================

DATA_NAME = [
    'AGNEWS',
    'AMAZON',
    'DBPEDIA',
    'IMDB',
    'MEDDIA',
    'QQP',
    'SST2',
    'TLDR',
    'WMT',
    'YELP'
]


# ============================================================
# 3. Layers
# ============================================================

LAYERS = [
    8,
    12,
    16,
    20,
    24,
    28,
    32
]


# ============================================================
# 4. Calculate similarity for every layer
# ============================================================

for layer in LAYERS:

    print()
    print("=" * 70)
    print(f"Calculating Layer {layer}")
    print("=" * 70)

    # ========================================================
    # Load dataset vectors
    # ========================================================

    dataset_vectors = {}

    for name in DATA_NAME:

        path = os.path.join(
            ACTIVATION_DIR,
            f'{name}_layer{layer}_dataset.pt'
        )

        print(
            f'Loading {name}: {path}'
        )

        vector = torch.load(
            path,
            map_location='cpu'
        )

        # 强制 float32
        vector = vector.float()

        print(
            f'  shape: {vector.shape}'
        )

        print(
            f'  dtype: {vector.dtype}'
        )

        dataset_vectors[name] = vector


    # ========================================================
    # Stack
    # ========================================================

    matrix = torch.stack(
        [
            dataset_vectors[name]
            for name in DATA_NAME
        ]
    )

    print()
    print(
        "Matrix shape:",
        matrix.shape
    )

    print(
        "Matrix dtype:",
        matrix.dtype
    )


    # ========================================================
    # L2 normalization
    # ========================================================

    matrix = F.normalize(
        matrix,
        p=2,
        dim=1
    )


    # ========================================================
    # Cosine similarity
    # ========================================================

    similarity_matrix = (
        matrix @ matrix.T
    )

    print()
    print(
        f"Layer {layer} cosine similarity:"
    )

    print(
        similarity_matrix
    )


    # ========================================================
    # Cosine distance
    # ========================================================

    distance_matrix = (
        1 - similarity_matrix
    )

    print()
    print(
        f"Layer {layer} cosine distance:"
    )

    print(
        distance_matrix
    )


    # ========================================================
    # Save .pt
    # ========================================================

    torch.save(
        similarity_matrix,
        os.path.join(
            OUTPUT_DIR,
            f'cosine_similarity_layer{layer}.pt'
        )
    )

    torch.save(
        distance_matrix,
        os.path.join(
            OUTPUT_DIR,
            f'cosine_distance_layer{layer}.pt'
        )
    )


    # ========================================================
    # Save similarity txt
    # ========================================================

    similarity_txt = os.path.join(
        OUTPUT_DIR,
        f'cosine_similarity_layer{layer}.txt'
    )

    with open(
        similarity_txt,
        'w',
        encoding='utf-8'
    ) as f:

        # header
        f.write(
            '\t'
            + '\t'.join(DATA_NAME)
            + '\n'
        )

        # rows
        for i, name in enumerate(DATA_NAME):

            values = (
                similarity_matrix[i]
                .tolist()
            )

            f.write(
                name
                + '\t'
                + '\t'.join(
                    f'{value:.6f}'
                    for value in values
                )
                + '\n'
            )


    # ========================================================
    # Save distance txt
    # ========================================================

    distance_txt = os.path.join(
        OUTPUT_DIR,
        f'cosine_distance_layer{layer}.txt'
    )

    with open(
        distance_txt,
        'w',
        encoding='utf-8'
    ) as f:

        # header
        f.write(
            '\t'
            + '\t'.join(DATA_NAME)
            + '\n'
        )

        # rows
        for i, name in enumerate(DATA_NAME):

            values = (
                distance_matrix[i]
                .tolist()
            )

            f.write(
                name
                + '\t'
                + '\t'.join(
                    f'{value:.6f}'
                    for value in values
                )
                + '\n'
            )


    print()
    print(
        f"Layer {layer} finished."
    )


# ============================================================
# 5. Finished
# ============================================================

print()
print("=" * 70)
print("All layers finished.")
print("=" * 70)

print(
    f"Results saved to: {OUTPUT_DIR}"
)