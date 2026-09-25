package com.dwu.fomorecorder

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import java.io.File

class MainActivity : Activity() {
    private lateinit var db: FomoRecorderDb
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        db = FomoRecorderDb(this)

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(36, 48, 36, 48)
        }

        content.addView(TextView(this).apply {
            text = "Fomo Local Recorder"
            textSize = 24f
        })

        content.addView(TextView(this).apply {
            text = "Records every family.fomo.app notification whose selected text contains the word bought or sold. Data stays on this Android device unless you explicitly export it."
            textSize = 16f
            setPadding(0, 24, 0, 24)
        })

        status = TextView(this).apply {
            textSize = 16f
            setPadding(0, 8, 0, 24)
        }
        content.addView(status)

        content.addView(Button(this).apply {
            text = "Open notification access"
            setOnClickListener {
                startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
            }
        }, matchWidth())

        content.addView(Button(this).apply {
            text = "Refresh local counts"
            setOnClickListener { refreshStats() }
        }, matchWidth())

        content.addView(Button(this).apply {
            text = "Export all rows to CSV"
            setOnClickListener { exportCsv() }
        }, matchWidth())

        content.addView(TextView(this).apply {
            text = "SQLite: ${getDatabasePath(FomoRecorderDb.DB_NAME).absolutePath}\nCSV exports: ${exportDirectory().absolutePath}"
            textSize = 13f
            setPadding(0, 24, 0, 0)
        })

        setContentView(ScrollView(this).apply { addView(content) })
    }

    override fun onResume() {
        super.onResume()
        refreshStats()
    }

    override fun onDestroy() {
        db.close()
        super.onDestroy()
    }

    private fun refreshStats() {
        Thread {
            val stats = db.stats()
            runOnUiThread {
                status.text = buildString {
                    appendLine("Recorded: ${stats.total}")
                    appendLine("Buys: ${stats.buys}")
                    appendLine("Sells: ${stats.sells}")
                    append("Latest: ${stats.latest ?: "none yet"}")
                }
            }
        }.start()
    }

    private fun exportCsv() {
        val destination = File(exportDirectory(), "fomo_buy_sell_events.csv")
        Thread {
            runCatching { db.exportCsv(destination) }
                .onSuccess { rows ->
                    runOnUiThread {
                        Toast.makeText(
                            this,
                            "Exported $rows rows to ${destination.absolutePath}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
                .onFailure { error ->
                    runOnUiThread {
                        Toast.makeText(
                            this,
                            "Export failed: ${error.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
        }.start()
    }

    private fun exportDirectory(): File =
        getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: File(filesDir, "exports")

    private fun matchWidth(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            topMargin = 12
        }
}
