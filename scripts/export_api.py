import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_models import build_pydantic_json_schemas
from main import create_app


def main() -> None:
    app = create_app()
    root = Path(".")

    openapi_path = root / "openapi.json"
    schema_path = root / "schema.json"

    openapi_path.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(build_pydantic_json_schemas(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"generated: {openapi_path}")
    print(f"generated: {schema_path}")


if __name__ == "__main__":
    main()
