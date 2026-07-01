import html

class HenriSDUIGenerator:
    """
    Server-Driven Accessible UI (SDUI) Generator:
    Translates abstract component models into WCAG 2.2 compliant structural HTML/JSON definitions,
    guaranteeing standard focus orders, label associations, and tabular layouts.
    """
    def __init__(self):
        pass

    def compile_form(self, form_id: str, fields: list) -> str:
        """
        Compiles form definition into WCAG-compliant HTML.
        Enforces label for/id association (3.3.2), aria-required (1.3.1), and aria-describedby.
        """
        form_html = [f'<form id="{html.escape(form_id)}" class="sdui-form">']
        
        for field in fields:
            f_id = html.escape(field.get("id", ""))
            f_type = html.escape(field.get("type", "text"))
            label_text = html.escape(field.get("label", ""))
            placeholder = html.escape(field.get("placeholder", ""))
            is_required = field.get("required", False)
            helper_text = html.escape(field.get("helper_text", ""))
            is_invalid = field.get("invalid", False)
            
            # Form field wrapper
            form_html.append('  <div class="form-group">')
            
            # Explicit Label Association (WCAG 3.3.2)
            req_indicator = ' <span class="required-indicator" aria-hidden="true">*</span>' if is_required else ''
            form_html.append(f'    <label for="{f_id}">{label_text}{req_indicator}</label>')
            
            # Helper text container with unique ID
            desc_id = f"{f_id}_helper"
            
            # Input tag compiling attributes
            req_attr = ' aria-required="true" required' if is_required else ''
            desc_attr = f' aria-describedby="{desc_id}"' if helper_text else ''
            inv_attr = ' aria-invalid="true"' if is_invalid else ''
            
            form_html.append(f'    <input id="{f_id}" type="{f_type}" placeholder="{placeholder}"{req_attr}{desc_attr}{inv_attr}>')
            
            # Programmatic Helper Description (WCAG 1.3.1)
            if helper_text:
                form_html.append(f'    <span id="{desc_id}" class="form-helper">{helper_text}</span>')
                
            form_html.append('  </div>')
            
        form_html.append('</form>')
        return "\n".join(form_html)

    def compile_table(self, table_id: str, caption: str, headers: list, rows: list) -> str:
        """
        Compiles data grid into WCAG-compliant Table wrapped in a scrollable area.
        Enforces Caption (1.3.1), Scope headers (1.3.1), and scrollable keyboard region (tabindex="0").
        """
        t_id = html.escape(table_id)
        caption_id = f"{t_id}_caption"
        
        # Scrollable Area Accessibility wrapper (tabindex="0" is critical for keyboard scrolling)
        table_html = [
            f'<div role="region" aria-labelledby="{caption_id}" tabindex="0" class="table-scroll-wrapper">',
            f'  <table id="{t_id}">'
        ]
        
        # Caption tag (WCAG 1.3.1)
        table_html.append(f'    <caption id="{caption_id}">{html.escape(caption)}</caption>')
        
        # Header structural sectioning (thead)
        table_html.append('    <thead>')
        table_html.append('      <tr>')
        for header in headers:
            # Header scope association (WCAG 1.3.1)
            table_html.append(f'        <th scope="col">{html.escape(header)}</th>')
        table_html.append('      </tr>')
        table_html.append('    </thead>')
        
        # Main data sectioning (tbody)
        table_html.append('    <tbody>')
        for row in rows:
            table_html.append('      <tr>')
            for idx, cell in enumerate(row):
                cell_text = str(cell)
                # Handle empty cells securely (wrap screen-reader-only context)
                if not cell_text.strip():
                    cell_text = '<span class="sr-only">Not applicable</span>-'
                
                # Check if first column acts as a row header
                if idx == 0:
                    table_html.append(f'        <th scope="row">{cell_text}</th>')
                else:
                    table_html.append(f'        <td>{cell_text}</td>')
            table_html.append('      </tr>')
        table_html.append('    </tbody>')
        
        table_html.append('  </table>')
        table_html.append('</div>')
        return "\n".join(table_html)

    def compile_card(self, card_id: str, title: str, cta_text: str, link_url: str, img_url: str, img_alt: str) -> str:
        """
        Compiles Card element.
        Enforces separate image alt, tabindex="-1" focus filters, and single title action (2.1.1).
        """
        c_id = html.escape(card_id)
        t_text = html.escape(title)
        cta = html.escape(cta_text)
        url = html.escape(link_url)
        img = html.escape(img_url)
        alt = html.escape(img_alt)
        
        card_html = [
            f'<div id="{c_id}" class="sdui-card">',
            # Redundant focus trap prevention: Image points to URL but is removed from keyboard focus
            f'  <a href="{url}" tabindex="-1" aria-hidden="true" class="card-image-link">',
            f'    <img src="{img}" alt="{alt}">',
            '  </a>',
            '  <div class="card-content">',
            # Focal Action Area: Title is the primary keyboard focus link
            f'    <h3 class="card-title"><a href="{url}">{t_text}</a></h3>',
            f'    <div class="card-actions">',
            # CTA button is also focusable
            f'      <a href="{url}" class="btn-card-cta">{cta}</a>',
            '    </div>',
            '  </div>',
            '</div>'
        ]
        return "\n".join(card_html)
