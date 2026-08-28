from flask import Flask, request, jsonify
import hashlib
import json
import math

app = Flask(__name__)

MAX_SAFE_INTEGER = 2**53 - 1

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_EXTENSIONS = {
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def is_nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def is_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def is_positive_safe_integer(value):
    return (
        is_safe_integer(value)
        and value > 0
    )


def is_finite_01(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def utf8_sort(value):
    return value.encode("utf-8")


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(
        text.encode("utf-8")
    )


def sha256_json(value):
    return sha256_text(
        compact_json(value)
    )


def add_violation(violations, code):
    violations.add(code)


def sorted_violations(violations):
    return sorted(
        violations,
        key=utf8_sort,
    )


# ============================================================
# JSON FILE PARSING
# ============================================================

def parse_json_file(files, filename, violations):
    """
    Parse a UTF-8 JSON file.

    Returns:
        parsed object
        or None
    """

    if filename not in files:
        return None

    text = files[filename]

    if not isinstance(text, str):
        add_violation(
            violations,
            f"INVALID_FILE:{filename}",
        )
        return None

    try:
        return json.loads(text)
    except Exception:
        add_violation(
            violations,
            f"INVALID_JSON:{filename}",
        )
        return None


# ============================================================
# INVENTORY
# ============================================================

def recompute_inventory(files):
    """
    Inventory contains every file except inventory.json.

    Sorted by UTF-8 filename.

    Each entry has exact key order:
        name, bytes, sha256
    """

    inventory = []

    for filename in sorted(
        (
            name
            for name in files.keys()
            if name != "inventory.json"
        ),
        key=utf8_sort,
    ):

        content = files[filename]

        if not isinstance(content, str):
            return None

        try:
            raw = content.encode("utf-8")
        except UnicodeEncodeError:
            return None

        inventory.append({
            "name": filename,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        })

    return inventory


def validate_inventory(
    files,
    violations,
):
    """
    Validate inventory.json against the actual files.
    """

    if "inventory.json" not in files:
        return None

    inventory_obj = parse_json_file(
        files,
        "inventory.json",
        violations,
    )

    if inventory_obj is None:
        return None

    if not isinstance(
        inventory_obj,
        list,
    ):
        add_violation(
            violations,
            "INVALID_FILE:inventory.json",
        )
        return None

    actual = recompute_inventory(
        files
    )

    if actual is None:
        add_violation(
            violations,
            "INVALID_FILE:inventory.json",
        )
        return None

    # Compare exact inventory structure.
    if inventory_obj != actual:
        add_violation(
            violations,
            "INVENTORY_MISMATCH",
        )

    # Detect extra files explicitly.
    tracked_names = {
        entry.get("name")
        for entry in inventory_obj
        if isinstance(entry, dict)
    }

    actual_names = set(
        name
        for name in files
        if name != "inventory.json"
    )

    if tracked_names != actual_names:
        add_violation(
            violations,
            "UNTRACKED_FILE",
        )

    return actual


# ============================================================
# ADAPTER CONFIG
# ============================================================

def validate_adapter_config(
    files,
    violations,
):
    config = parse_json_file(
        files,
        "adapter_config.json",
        violations,
    )

    if config is None:
        return None

    valid = True

    if not isinstance(config, dict):
        valid = False

    else:

        if not is_positive_safe_integer(
            config.get("r")
        ):
            valid = False

        target_modules = config.get(
            "target_modules"
        )

        if not isinstance(
            target_modules,
            list,
        ):
            valid = False

        else:

            if len(target_modules) == 0:
                valid = False

            seen = set()

            for module in target_modules:

                if not is_nonempty_string(
                    module
                ):
                    valid = False
                    continue

                if module in seen:
                    valid = False

                seen.add(module)

    if not valid:
        add_violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

    return config


# ============================================================
# TRAINING MANIFEST
# ============================================================

MANIFEST_FIELDS = [
    "baseRevision",
    "task",
    "datasetDigest",
    "codeDigest",
    "trainingConfigDigest",
    "modelArtifactDigest",
    "evaluationArtifactDigest",
]


def validate_training_manifest(
    files,
    violations,
):
    manifest = parse_json_file(
        files,
        "training_manifest.json",
        violations,
    )

    if manifest is None:
        return None

    valid = True

    if not isinstance(
        manifest,
        dict,
    ):
        valid = False

    else:

        base_revision = manifest.get(
            "baseRevision"
        )

        if (
            not isinstance(
                base_revision,
                str,
            )
            or len(base_revision) != 40
            or any(
                c not in "0123456789abcdef"
                for c in base_revision
            )
        ):
            add_violation(
                violations,
                "MUTABLE_BASE_REVISION",
            )
            valid = False

        for field in MANIFEST_FIELDS[1:]:

            if not is_nonempty_string(
                manifest.get(field)
            ):

                add_violation(
                    violations,
                    f"MISSING_MANIFEST_FIELD:{field}",
                )

                valid = False

    if not valid:
        add_violation(
            violations,
            "INVALID_TRAINING_MANIFEST",
        )

    return manifest


# ============================================================
# ARTIFACT DIGESTS
# ============================================================

def validate_artifacts(
    files,
    manifest,
    evaluation,
    violations,
):
    model_digest = None
    evaluation_digest = None

    if (
        "adapter_model.safetensors"
        in files
    ):

        model_bytes = files[
            "adapter_model.safetensors"
        ].encode("utf-8")

        model_digest = sha256_bytes(
            model_bytes
        )

        if (
            isinstance(manifest, dict)
            and is_nonempty_string(
                manifest.get(
                    "modelArtifactDigest"
                )
            )
            and manifest[
                "modelArtifactDigest"
            ] != model_digest
        ):

            add_violation(
                violations,
                "MODEL_ARTIFACT_MISMATCH",
            )

    if "evaluation.json" in files:

        evaluation_bytes = files[
            "evaluation.json"
        ].encode("utf-8")

        evaluation_digest = sha256_bytes(
            evaluation_bytes
        )

        if (
            isinstance(manifest, dict)
            and is_nonempty_string(
                manifest.get(
                    "evaluationArtifactDigest"
                )
            )
            and manifest[
                "evaluationArtifactDigest"
            ] != evaluation_digest
        ):

            add_violation(
                violations,
                "EVALUATION_ARTIFACT_MISMATCH",
            )

    return (
        model_digest,
        evaluation_digest,
    )


# ============================================================
# EVALUATION
# ============================================================

def validate_evaluation(
    files,
    manifest,
    model_digest,
    required_slices,
    violations,
):
    evaluation = parse_json_file(
        files,
        "evaluation.json",
        violations,
    )

    if evaluation is None:
        return None

    valid = True

    if not isinstance(
        evaluation,
        dict,
    ):
        add_violation(
            violations,
            "INVALID_EVALUATION",
        )
        return None

    # --------------------------------------------------------
    # Model digest binding.
    #
    # Accept the model digest under the expected
    # modelArtifactDigest field.
    # --------------------------------------------------------

    evaluation_model_digest = (
        evaluation.get(
            "modelArtifactDigest"
        )
    )

    if (
        not is_nonempty_string(
            evaluation_model_digest
        )
        or evaluation_model_digest
        != model_digest
    ):

        add_violation(
            violations,
            "EVALUATION_DIGEST_MISMATCH",
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    aggregate = evaluation.get(
        "aggregate"
    )

    if not is_finite_01(aggregate):

        add_violation(
            violations,
            "INVALID_AGGREGATE",
        )

    # --------------------------------------------------------
    # Required slices
    # --------------------------------------------------------

    slices = evaluation.get(
        "slices"
    )

    if not isinstance(
        slices,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_EVALUATION",
        )

    else:

        for slice_name in required_slices:

            if slice_name not in slices:

                add_violation(
                    violations,
                    f"MISSING_SLICE:{slice_name}",
                )

            elif not is_finite_01(
                slices[slice_name]
            ):

                add_violation(
                    violations,
                    f"SLICE_RANGE:{slice_name}",
                )

    return evaluation


# ============================================================
# MODEL CARD
# ============================================================

MODEL_CARD_PREFIX = (
    '<!-- tds-model-card '
)


def extract_model_card_markers(
    readme
):
    """
    Find complete markers.

    Braces inside JSON strings are treated as ordinary
    characters because we only use the literal --> delimiter.
    """

    markers = []

    position = 0

    while True:

        start = readme.find(
            MODEL_CARD_PREFIX,
            position,
        )

        if start == -1:
            break

        end = readme.find(
            "-->",
            start + len(
                MODEL_CARD_PREFIX
            ),
        )

        if end == -1:
            # Unterminated marker.
            markers.append(
                readme[
                    start + len(
                        MODEL_CARD_PREFIX
                    ):
                ]
            )
            break

        payload = readme[
            start + len(
                MODEL_CARD_PREFIX
            ):end
        ]

        markers.append(payload)

        position = end + 3

    return markers


def validate_model_card(
    files,
    manifest,
    evaluation,
    policy,
    violations,
):
    readme = files.get(
        "README.md"
    )

    if not isinstance(
        readme,
        str,
    ):

        return

    markers = extract_model_card_markers(
        readme
    )

    if len(markers) == 0:

        add_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        add_violation(
            violations,
            "MISSING_MODEL_CARD",
        )

        return

    if len(markers) > 1:

        add_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        return

    payload = markers[0]

    try:
        card = json.loads(payload)
    except Exception:

        add_violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    if not isinstance(card, dict):

        add_violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    # Required consistency.
    expected = {
        "task": (
            manifest.get("task")
            if isinstance(
                manifest,
                dict,
            )
            else None
        ),
        "baseRevision": (
            manifest.get(
                "baseRevision"
            )
            if isinstance(
                manifest,
                dict,
            )
            else None
        ),
        "datasetDigest": (
            manifest.get(
                "datasetDigest"
            )
            if isinstance(
                manifest,
                dict,
            )
            else None
        ),
        "modelArtifactDigest": (
            manifest.get(
                "modelArtifactDigest"
            )
            if isinstance(
                manifest,
                dict,
            )
            else None
        ),
        "license": policy.get(
            "license"
        ),
        "intendedUse": policy.get(
            "intendedUse"
        ),
        "limitations": policy.get(
            "limitations"
        ),
    }

    mismatch = False

    for field, expected_value in (
        expected.items()
    ):

        if card.get(field) != expected_value:
            mismatch = True

    if mismatch:

        add_violation(
            violations,
            "MODEL_CARD_MISMATCH",
        )


# ============================================================
# POLICY
# ============================================================

def validate_policy(
    policy,
    violations,
):
    if not isinstance(
        policy,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_POLICY",
        )

        return []

    required_slices = policy.get(
        "requiredSlices"
    )

    if not isinstance(
        required_slices,
        list,
    ) or len(required_slices) == 0:

        add_violation(
            violations,
            "INVALID_POLICY",
        )

        return []

    seen = set()

    for name in required_slices:

        if not is_nonempty_string(
            name
        ):

            add_violation(
                violations,
                "INVALID_POLICY",
            )

        elif name in seen:

            add_violation(
                violations,
                "INVALID_POLICY",
            )

        seen.add(name)

    for field in (
        "license",
        "intendedUse",
        "limitations",
    ):

        if not is_nonempty_string(
            policy.get(field)
        ):

            add_violation(
                violations,
                "INVALID_POLICY",
            )

    return required_slices


# ============================================================
# MAIN VERIFICATION
# ============================================================

def verify_bundle(body):

    violations = set()

    policy = body.get(
        "policy"
    )

    files = body.get(
        "files"
    )

    required_slices = validate_policy(
        policy,
        violations,
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    if not isinstance(
        files,
        dict,
    ):

        # Endpoint-level invalid input is handled before
        # this function.
        add_violation(
            violations,
            "INVALID_FILE:files",
        )

        return {
            "decision": "reject",
            "violations":
                sorted_violations(
                    violations
                ),
            "inventoryDigest": None,
        }

    # --------------------------------------------------------
    # Missing required files
    # --------------------------------------------------------

    for filename in REQUIRED_FILES:

        if filename not in files:

            add_violation(
                violations,
                f"MISSING_FILE:{filename}",
            )

    # --------------------------------------------------------
    # Extra / unsafe files
    # --------------------------------------------------------

    for filename in files:

        if not isinstance(
            filename,
            str,
        ):
            add_violation(
                violations,
                "INVALID_FILE:"
            )
            continue

        if filename not in REQUIRED_FILES:

            add_violation(
                violations,
                f"UNTRACKED_FILE:{filename}",
            )

        lower = filename.lower()

        for extension in UNSAFE_EXTENSIONS:

            if lower.endswith(
                extension
            ):

                add_violation(
                    violations,
                    "UNSAFE_WEIGHTS",
                )

                break

    # --------------------------------------------------------
    # File values must be strings.
    # --------------------------------------------------------

    for filename, content in files.items():

        if not isinstance(
            filename,
            str,
        ):

            continue

        if not isinstance(
            content,
            str,
        ):

            add_violation(
                violations,
                f"INVALID_FILE:{filename}",
            )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    actual_inventory = (
        validate_inventory(
            files,
            violations,
        )
    )

    if actual_inventory is None:

        inventory_digest = None

    else:

        inventory_digest = sha256_json(
            actual_inventory
        )

    # --------------------------------------------------------
    # Adapter config
    # --------------------------------------------------------

    adapter_config = (
        validate_adapter_config(
            files,
            violations,
        )
    )

    # --------------------------------------------------------
    # Training manifest
    # --------------------------------------------------------

    manifest = (
        validate_training_manifest(
            files,
            violations,
        )
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation = None

    model_digest = None

    if (
        "adapter_model.safetensors"
        in files
    ):

        try:

            model_digest = sha256_bytes(
                files[
                    "adapter_model.safetensors"
                ].encode("utf-8")
            )

        except Exception:

            add_violation(
                violations,
                "INVALID_FILE:"
                "adapter_model.safetensors",
            )

    evaluation_digest = None

    if "evaluation.json" in files:

        try:

            evaluation_digest = sha256_bytes(
                files[
                    "evaluation.json"
                ].encode("utf-8")
            )

        except Exception:

            add_violation(
                violations,
                "INVALID_FILE:"
                "evaluation.json",
            )

    # Manifest artifact checks.
    if isinstance(manifest, dict):

        if (
            model_digest is not None
            and manifest.get(
                "modelArtifactDigest"
            )
            != model_digest
        ):

            add_violation(
                violations,
                "MODEL_ARTIFACT_MISMATCH",
            )

        if (
            evaluation_digest is not None
            and manifest.get(
                "evaluationArtifactDigest"
            )
            != evaluation_digest
        ):

            add_violation(
                violations,
                "EVALUATION_ARTIFACT_MISMATCH",
            )

    # --------------------------------------------------------
    # Evaluation semantic validation
    # --------------------------------------------------------

    if (
        "evaluation.json" in files
    ):

        evaluation = validate_evaluation(
            files,
            manifest,
            model_digest,
            required_slices,
            violations,
        )

    # --------------------------------------------------------
    # Model card
    # --------------------------------------------------------

    validate_model_card(
        files,
        manifest,
        evaluation,
        policy,
        violations,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    final_violations = (
        sorted_violations(
            violations
        )
    )

    decision = (
        "admit"
        if len(final_violations) == 0
        else "reject"
    )

    return {
        "decision": decision,
        "violations": final_violations,
        "inventoryDigest": inventory_digest,
    }


# ============================================================
# HTTP ENDPOINT
# ============================================================

@app.route(
    "/verify-bundle",
    methods=["POST"],
)
def verify_bundle_endpoint():

    if not request.is_json:
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    try:
        body = request.get_json()
    except Exception:
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    if not isinstance(
        body,
        dict,
    ):

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # Missing/non-object policy or files
    # is an HTTP 400 condition.
    if (
        "policy" not in body
        or not isinstance(
            body["policy"],
            dict,
        )
    ):

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    if (
        "files" not in body
        or not isinstance(
            body["files"],
            dict,
        )
    ):

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    result = verify_bundle(
        body
    )

    return jsonify(result), 200


@app.route("/", methods=["GET"])
def home():
    return "verify-bundle service running", 200


if __name__ == "__main__":

    import os

    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
