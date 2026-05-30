# =============================================================
# services/early_warning_service.py
# Phase 6 — Early Warning Report Service
#
# FIXES APPLIED:
#   ✅ FIX 1: Fetches from early_warning_reports table
#   ✅ FIX 2: Filters High + Critical risk students correctly
#   ✅ FIX 3: Returns full prediction metadata
#   ✅ FIX 4: Graceful handling when table is empty
#   ✅ FIX 5: Supabase offline protection
#   ✅ FIX 6: Hugging Face compatible
#   ✅ FIX 7: Clean API response structure
#   ✅ FIX 8: Proper advisory notes for admin users
#   ✅ FIX 9: Supports academic_year filtering
#   ✅ FIX 10: Safe limit handling
# =============================================================

from database.supabase_client import supabase


# =============================================================
# EARLY WARNING REPORT SERVICE
# =============================================================

def get_early_warning_report(
    academic_year: str = "",
    limit: int = 5000,
) -> dict:
    """
    Fetch High and Critical risk students
    from early_warning_reports table.

    IMPORTANT:
    This table is populated by:
        POST /api/v6/batch-inference

    If no data exists:
        Run batch-inference first.
    """

    try:

        # =====================================================
        # CHECK SUPABASE CONNECTION
        # =====================================================

        if supabase is None:

            return {
                "status": "error",

                "message": (
                    "Supabase not configured. "
                    "Please set:\n"
                    "- SUPABASE_URL\n"
                    "- SUPABASE_KEY\n"
                    "inside .env or Hugging Face Secrets."
                ),
            }

        # =====================================================
        # SAFE LIMIT
        # =====================================================

        try:
            limit = int(limit)

            if limit <= 0:
                limit = 5000

        except Exception:
            limit = 5000

        # =====================================================
        # BUILD QUERY
        # =====================================================

        query = (
            supabase
            .table("early_warning_reports")
            .select("*")
        )

        # =====================================================
        # OPTIONAL ACADEMIC YEAR FILTER
        # =====================================================

        if academic_year:

            query = query.eq(
                "academic_year",
                str(academic_year)
            )

        # =====================================================
        # FILTER ONLY HIGH + CRITICAL
        # =====================================================

        query = (
            query
            .in_(
                "risk_level",
                [
                    "High",
                    "Critical",
                ]
            )
            .limit(limit)
        )

        # =====================================================
        # EXECUTE QUERY
        # =====================================================

        response = query.execute()

        rows = response.data or []

        # =====================================================
        # NO DATA FOUND
        # =====================================================

        if not rows:

            return {

                "status": "success",

                "academic_year": (
                    academic_year
                    or "all"
                ),

                "total_records": 0,

                "data": [],

                "advisory_note": (
                    "No High/Critical risk students found.\n\n"
                    "Please run:\n"
                    "POST /api/v6/batch-inference\n\n"
                    "to generate ML prediction records first."
                ),
            }

        # =====================================================
        # SUCCESS RESPONSE
        # =====================================================

        return {

            "status": "success",

            "academic_year": (
                academic_year
                or "all"
            ),

            "total_records": len(rows),

            "data": rows,

            "advisory_note": (
                "These predictions are advisory only.\n"
                "No student services are automatically restricted.\n"
                "School administration must review and approve "
                "any recovery or intervention actions."
            ),
        }

    # =========================================================
    # GLOBAL ERROR HANDLER
    # =========================================================

    except Exception as error:

        import traceback

        traceback.print_exc()

        return {

            "status": "error",

            "message": str(error),
        }