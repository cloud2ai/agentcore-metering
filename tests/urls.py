"""
Root URLconf for tests: mount agentcore_tracking admin API under api/v1/admin/.
"""
from django.urls import path, include

urlpatterns = [
    path("api/v1/admin/", include("agentcore_tracking.adapters.django.urls")),
]
