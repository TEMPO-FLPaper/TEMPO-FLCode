"""Reference-derived reporting/figures_tables. See README.md."""

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class Operations:
    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def show_confusion_matrix_for_weights(self, weights, df, title, output_path, temperature=1.0):
        y_true, y_pred, _ = self.predictions_from_weights(weights, df, temperature)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.NUM_CLASSES))
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, aspect="auto")
        plt.xticks(range(self.NUM_CLASSES), self.CLASS_NAMES, rotation=35)
        plt.yticks(range(self.NUM_CLASSES), self.CLASS_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.colorbar(label="Samples")
        for i in range(self.NUM_CLASSES):
            for j in range(self.NUM_CLASSES):
                plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return cm

    # Reference cells (zero-based): {'ditto': 18, 'hfl': 18, 'flat': 18}
    def write_v12_2_metric_workbook__ditto_hfl_flat(self, path, tables, percent_columns):
        path = Path(path)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, dataframe in tables.items():
                dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        from openpyxl import load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill

        workbook = load_workbook(path)
        header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            if worksheet.max_row >= 1 and worksheet.max_column >= 1:
                worksheet.auto_filter.ref = worksheet.dimensions
            headers = {cell.column: cell.value for cell in worksheet[1]}
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for column_cells in worksheet.columns:
                column_letter = column_cells[0].column_letter
                max_length = max(
                    (len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                )
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 38)
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    header = headers.get(cell.column)
                    if header in percent_columns:
                        cell.number_format = "0.00%"
                    elif header in {
                        "test_samples",
                        "owned_test_samples",
                        "models",
                        "client_count",
                        "round",
                    }:
                        cell.number_format = "#,##0"
        workbook.save(path)
