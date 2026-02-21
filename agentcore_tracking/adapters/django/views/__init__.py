"""
Admin API views split by domain.
"""
from .config_catalog import (
    AdminLLMConfigModelsView,
    AdminLLMConfigProvidersView,
)
from .config_management import (
    AdminLLMConfigAllListView,
    AdminLLMConfigDetailView,
    AdminLLMConfigGlobalView,
    AdminLLMConfigUserDetailView,
    AdminLLMConfigUserListView,
)
from .config_validation import (
    AdminLLMConfigTestCallView,
    AdminLLMConfigTestView,
)
from .usage import AdminLLMUsageListView, AdminTokenStatsView
from .users import AdminUsersListView

__all__ = [
    "AdminLLMConfigAllListView",
    "AdminLLMConfigDetailView",
    "AdminLLMConfigGlobalView",
    "AdminLLMConfigModelsView",
    "AdminLLMConfigProvidersView",
    "AdminLLMConfigTestCallView",
    "AdminLLMConfigTestView",
    "AdminLLMConfigUserDetailView",
    "AdminLLMConfigUserListView",
    "AdminLLMUsageListView",
    "AdminTokenStatsView",
    "AdminUsersListView",
]
