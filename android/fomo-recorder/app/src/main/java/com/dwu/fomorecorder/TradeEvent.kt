package com.dwu.fomorecorder

data class TradeEvent(
    val packageName: String,
    val notificationKey: String,
    val notificationId: Int,
    val notificationTag: String?,
    val postTimeMs: Long,
    val capturedTimeMs: Long,
    val title: String,
    val text: String,
    val bigText: String?,
    val selectedText: String,
    val action: String,
    val trader: String?,
    val token: String?,
    val marketCapUsd: Double?,
    val sourceAmountUsd: Double?,
    val channelId: String?,
    val category: String?,
    val groupKey: String?,
    val hasContentIntent: Boolean,
    val notificationActionCount: Int,
)
