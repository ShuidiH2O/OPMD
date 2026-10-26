import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

def generate_table_v20(output_filename="table_v20.png"):
    fig, ax = plt.subplots(figsize=(11, 2.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('none')

    renderer = fig.canvas.get_renderer()

    x_method = 0.03
    col_x = [0.34, 0.46, 0.62, 0.76]

    y_top = 0.91
    y_mid_line = 0.80
    y_header_bot = 0.69

    y_h1 = 0.855
    y_h2 = 0.745
    y_method = (y_top + y_header_bot) / 2

    row_h = 0.075
    rows_y = [y_header_bot - 0.055 - i * row_h for i in range(6)]
    y_bottom = rows_y[-1] - 0.035

    math_line_start = col_x[0] - 0.075
    math_line_end = col_x[1] + 0.075
    minerva_line_start = col_x[2] - 0.075
    minerva_line_end = col_x[3] + 0.075

    ax.plot([0.02, minerva_line_end], [y_top, y_top], color='black', lw=1.5)
    ax.plot([math_line_start, math_line_end], [y_mid_line, y_mid_line], color='black', lw=0.8)
    ax.plot([minerva_line_start, minerva_line_end], [y_mid_line, y_mid_line], color='black', lw=0.8)
    ax.plot([0.02, minerva_line_end], [y_header_bot, y_header_bot], color='black', lw=0.8)
    ax.plot([0.02, minerva_line_end], [y_bottom, y_bottom], color='black', lw=1.5)

    ax.text(x_method, y_method, "Method", fontweight='bold', fontsize=10, va='center')

    math_center = (col_x[0] + col_x[1]) / 2
    ax.text(math_center, y_h1, "MATH-500", fontweight='bold', fontsize=10, ha='center', va='center')
    ax.text(col_x[0], y_h2, r"$\mathbf{pass@1}$", fontsize=9.5, ha='center', va='center')
    ax.text(col_x[1], y_h2, r"$\mathbf{avg@16}$", fontsize=9.5, ha='center', va='center')

    minerva_center = (col_x[2] + col_x[3]) / 2
    ax.text(minerva_center, y_h1, "Minerva", fontweight='bold', fontsize=10, ha='center', va='center')
    ax.text(col_x[2], y_h2, r"$\mathbf{pass@1}$", fontsize=9.5, ha='center', va='center')
    ax.text(col_x[3], y_h2, r"$\mathbf{avg@16}$", fontsize=9.5, ha='center', va='center')

    methods = [
        "Qwen3-1.7B-Base (Student)",
        "Qwen3-4B-Base (Teacher 1)",
        "Qwen3-14B-Base (Teacher 2)",
        "Student NTP",
        "Teacher 1 OPMD",
        "Teacher 2 OPMD"
    ]

    blue_bg = '#80D8FF'
    green_bg = '#B9F6CA'

    data = [
        [("76.6", "", "", None, None), ("18.53", "", "", None, None), ("30.5", "", "", None, None), ("4.66", "", "", None, None)],
        [("84.8", "", "", None, None), ("21.32", "", "", None, None), ("34.9", "", "", None, None), ("5.58", "", "", None, None)],
        [("94.2", "", "", None, None), ("68.64", "", "", None, None), ("51.5", "", "", None, None), ("24.03", "", "", None, None)],
        [
            ("18.2", "↓", "58.40", '#008000', blue_bg),
            ("1.30", "↓", "17.23", '#008000', blue_bg),
            ("7.4", "↓", "23.10", '#008000', blue_bg),
            ("0.60", "↓", "4.06", '#008000', blue_bg)
        ],
        [
            ("81.4", "↑", "4.80", '#D32F2F', green_bg),
            ("30.49", "↑", "11.96", '#D32F2F', green_bg),
            ("34.93", "↑", "4.43", '#D32F2F', green_bg),
            ("6.99", "↑", "2.33", '#D32F2F', green_bg)
        ],
        [
            ("82.4", "↑", "5.80", '#D32F2F', green_bg),
            ("29.90", "↑", "11.37", '#D32F2F', green_bg),
            ("36.03", "↑", "5.53", '#D32F2F', green_bg),
            ("8.02", "↑", "3.36", '#D32F2F', green_bg)
        ]
    ]

    for idx, (y_pos, row_vals) in enumerate(zip(rows_y, data)):
        m_name = methods[idx]
        if idx >= 3:
            ax.text(x_method, y_pos, m_name, fontsize=9.2, va='center', fontweight='bold')
        else:
            ax.text(x_method, y_pos, m_name, fontsize=9.2, va='center')

        for col_i, (val_str, arrow_str, diff_num_str, text_color, bg_color) in enumerate(row_vals):
            cx = col_x[col_i]

            if arrow_str:
                w_val = len(val_str) * 0.0058
                w_arrow = 0.0050
                gap_internal = 0.004
                content_w = w_val + gap_internal + w_arrow
                padding_x = 0.0045
                box_w = content_w + 2 * padding_x
                gap_to_diff = 0.004
                w_diff = len(diff_num_str) * 0.0055
                total_group_w = box_w + gap_to_diff + w_diff
                x_start = cx - (total_group_w / 2.0)
                box_x0 = x_start

                val_x = box_x0 + padding_x + (w_val / 2.0)
                arrow_x = box_x0 + padding_x + w_val + gap_internal + (w_arrow / 2.0)
                diff_x = box_x0 + box_w + gap_to_diff

                t_val = ax.text(val_x, y_pos, val_str, color='black', fontweight='bold', fontsize=9.5,
                                ha='center', va='center', zorder=3)
                t_arrow = ax.text(arrow_x, y_pos, arrow_str, color=text_color, fontweight='bold', fontsize=9.5,
                                  ha='center', va='center', zorder=3)
                t_diff = ax.text(diff_x, y_pos, diff_num_str, color=text_color, fontweight='bold', fontsize=9.0,
                                 ha='left', va='center', zorder=3)

                b1 = t_val.get_window_extent(renderer)
                b2 = t_arrow.get_window_extent(renderer)
                pad_px = 3.0
                x0 = min(b1.x0, b2.x0) - pad_px
                x1 = max(b1.x1, b2.x1) + pad_px
                y0 = min(b1.y0, b2.y0) - pad_px
                y1 = max(b1.y1, b2.y1) + pad_px
                inv = ax.transData.inverted()
                (dx0, dy0) = inv.transform((x0, y0))
                (dx1, dy1) = inv.transform((x1, y1))
                rect = patches.FancyBboxPatch(
                    (dx0, dy0), dx1 - dx0, dy1 - dy0,
                    boxstyle="round,pad=0,rounding_size=0.004",
                    facecolor=bg_color, edgecolor='none', alpha=0.45, zorder=1
                )
                ax.add_patch(rect)
            else:
                ax.text(cx, y_pos, val_str, color='black', fontweight='normal', fontsize=9.2, ha='center', va='center')

    caption_text = (
        "Table: Evaluation results on MATH-500 and Minerva (n=16 sampling).\n"
        "Highlighted values and arrows show performance gain/loss relative to Qwen3-1.7B-Base (Student)."
    )
    ax.text(0.02, y_bottom - 0.04, caption_text, fontsize=8, va='top', ha='left')

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.02, dpi=300, facecolor='white')
    pdf_name = output_filename.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_name, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close()

generate_table_v20()