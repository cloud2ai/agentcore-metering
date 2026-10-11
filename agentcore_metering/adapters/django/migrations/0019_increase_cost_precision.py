from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "agentcore_metering",
            "0018_alter_llmusageseries_options_and_more",
        ),
    ]

    operations = [
        migrations.AlterField(
            model_name="llmusage",
            name="cost",
            field=models.DecimalField(
                max_digits=18,
                decimal_places=12,
                null=True,
                blank=True,
                default=None,
                help_text=(
                    "Cost amount (use with cost_currency for multi-currency)"
                ),
            ),
        ),
        migrations.AlterField(
            model_name="llmusageseries",
            name="total_cost",
            field=models.DecimalField(
                max_digits=18,
                decimal_places=12,
                null=True,
                blank=True,
            ),
        ),
    ]
