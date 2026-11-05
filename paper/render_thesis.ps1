# Update the table of contents and page fields of the thesis draft in Word, save it, and export a PDF.
#   powershell -ExecutionPolicy Bypass -File paper\render_thesis.ps1 -Docx <path.docx> -Pdf <path.pdf>
param([string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($Docx)
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
  $doc.Fields.Update() | Out-Null
  $doc.Save()
  if ($Pdf) { $doc.ExportAsFixedFormat($Pdf, 17) }
  "pages: " + $doc.ComputeStatistics(2)
  $doc.Close([ref]0)
} finally {
  $word.Quit([ref]0)
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
  [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
