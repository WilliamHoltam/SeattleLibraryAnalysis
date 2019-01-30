dbutils.fs.put("dbfs:/databricks/init/sf-initiation.sh" ,"""
#!/bin/bash
/databricks/python/bin/pip install --upgrade pip
/databricks/python/bin/pip uninstall pyopenssl -y
/databricks/python/bin/pip install --upgrade pyopenssl
/databricks/python/bin/pip install --upgrade snowflake-sqlalchemy
/databricks/python/bin/pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 sudo -H pip install -U -y
""", True)
