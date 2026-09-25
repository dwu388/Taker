package com.dwu.fomorecorder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TradeParserTest {
    @Test
    fun parsesBuyWithKMarketCap() {
        val parsed = TradeParser.parse(
            title = "BONK at $47.3k MC",
            selectedText = "@cryptojuggler3 bought $837"
        )!!

        assertEquals("bought", parsed.action)
        assertEquals("cryptojuggler3", parsed.trader)
        assertEquals("BONK", parsed.token)
        assertEquals(47_300.0, parsed.marketCapUsd!!, 0.001)
        assertEquals(837.0, parsed.sourceAmountUsd!!, 0.001)
    }

    @Test
    fun parsesSellWithCommaAmountAndMillionMarketCap() {
        val parsed = TradeParser.parse(
            title = "WIF at $2.4M market cap",
            selectedText = "@pointfarmcap sold $1,250.50"
        )!!

        assertEquals("sold", parsed.action)
        assertEquals("pointfarmcap", parsed.trader)
        assertEquals("WIF", parsed.token)
        assertEquals(2_400_000.0, parsed.marketCapUsd!!, 0.001)
        assertEquals(1_250.50, parsed.sourceAmountUsd!!, 0.001)
    }

    @Test
    fun ignoresNonTradeTextAndWordFragments() {
        assertNull(TradeParser.parse("BONK at $47k MC", "new thesis posted"))
        assertNull(TradeParser.parse("BONK at $47k MC", "market is oversold"))
    }
}
