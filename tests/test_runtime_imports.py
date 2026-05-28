import importlib
import os
import unittest


class RuntimeImportTests(unittest.TestCase):
    def test_imports_do_not_require_env_vars(self):
        backup = {k: os.environ.get(k) for k in ["CITY", "DAGSHUB_USERNAME", "DAGSHUB_REPO", "DAGSHUB_TOKEN"]}
        try:
            for key in backup:
                os.environ.pop(key, None)
            import src.utils
            import src.predict
            import src.feature_pipeline
            importlib.reload(src.utils)
            importlib.reload(src.predict)
            importlib.reload(src.feature_pipeline)
        finally:
            for key, value in backup.items():
                if value is not None:
                    os.environ[key] = value

    def test_city_validation_runtime(self):
        from src.utils import get_city_config

        current = os.environ.get("CITY")
        try:
            os.environ["CITY"] = "lahore"
            with self.assertRaises(ValueError):
                get_city_config()

            os.environ["CITY"] = "karachi"
            city, lat, lon = get_city_config()
            self.assertEqual(city, "karachi")
            self.assertAlmostEqual(lat, 24.8607)
            self.assertAlmostEqual(lon, 67.0011)
        finally:
            if current is None:
                os.environ.pop("CITY", None)
            else:
                os.environ["CITY"] = current


if __name__ == "__main__":
    unittest.main()
