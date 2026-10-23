from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("agentcore_metering", "0019_increase_cost_precision"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmconfig",
            name="fallback",
            field=models.JSONField(
                null=True,
                blank=True,
                default=None,
                help_text=(
                    "Ordered backup models and bounded failover policy."
                ),
            ),
        ),
    ]
