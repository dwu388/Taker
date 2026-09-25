package com.dwu.fomorecorder

object TradeParser {
    data class ParsedTrade(
        val action: String,
        val trader: String?,
        val token: String?,
        val marketCapUsd: Double?,
        val sourceAmountUsd: Double?,
    )

    private val actionRegex = Regex("""(?i)\b(bought|sold)\b""")
    private val traderRegex = Regex("""@([A-Za-z0-9_.-]+)""")
    private val moneyRegex = Regex("""(?i)\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kmb])?""")
    private val marketCapRegex = Regex(
        """(?i)\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kmb])?\s*(?:mc|market\s*cap)\b"""
    )
    private val tokenRegex = Regex("""(?i)^\s*(.+?)\s+at\s+\$""")

    fun parse(title: String, selectedText: String): ParsedTrade? {
        val actionMatch = actionRegex.find(selectedText) ?: return null
        val action = actionMatch.groupValues[1].lowercase()

        val trader = traderRegex.find(selectedText)?.groupValues?.get(1)
        val token = tokenRegex.find(title)?.groupValues?.get(1)?.trim()?.takeIf { it.isNotEmpty() }

        val marketCapUsd = marketCapRegex.find(title)?.let { parseMoneyMatch(it) }

        val afterAction = selectedText.substring(actionMatch.range.last + 1)
        val sourceAmountUsd = moneyRegex.find(afterAction)?.let { parseMoneyMatch(it) }

        return ParsedTrade(
            action = action,
            trader = trader,
            token = token,
            marketCapUsd = marketCapUsd,
            sourceAmountUsd = sourceAmountUsd,
        )
    }

    private fun parseMoneyMatch(match: MatchResult): Double? {
        val numeric = match.groupValues.getOrNull(1)
            ?.replace(",", "")
            ?.toDoubleOrNull()
            ?: return null
        val suffix = match.groupValues.getOrNull(2)?.lowercase().orEmpty()
        val multiplier = when (suffix) {
            "k" -> 1_000.0
            "m" -> 1_000_000.0
            "b" -> 1_000_000_000.0
            else -> 1.0
        }
        return numeric * multiplier
    }
}
