def construir_respuesta(hits):
    """
    Recibe una lista de dicts (o model_dump de Pydantic) con campos:
    regla, intento, accion, resultado, observacion, paso_a_paso, evidencia, score
    """
    if not hits:
        return "Sin resultados."

    # Intento principal: primer exitoso, si no hay usa el primero
    exitosos = [h for h in hits if (h.get('resultado', '').strip().lower() == 'exitoso')]
    principal = exitosos[0] if exitosos else hits[0]

    regla = (principal.get('regla') or 'general').strip() or 'general'
    paso_o_accion = principal.get('paso_a_paso') or principal.get('accion', '(sin paso_a_paso)')

    resumen = [
        f"✅ Configuración recomendada para **{regla}** (basado en intento exitoso #{int(principal.get('intento', 0))}):",
        paso_o_accion,
        ""
    ]

    # Top 3 más similares (con score) a modo de evidencia
    resumen.append("🔎 Resultados similares (top 3):")
    for h in hits[:3]:
        resumen.append(
            f"- #{int(h.get('intento',0))} [{h.get('regla','')} | {h.get('resultado','')} | score={h.get('score',0):.3f}] → {h.get('accion','')}"
        )

    # Fallidos relacionados (máx 3)
    fallidos = [
        h for h in hits
        if h is not principal and h.get('resultado', '').strip().lower() != 'exitoso'
    ]
    if fallidos:
        resumen.append("\n❌ Intentos fallidos relacionados (para no repetir errores):")
        for f in fallidos[:3]:
            obs = f.get('observacion') or f.get('resultado') or ''
            resumen.append(f"- Intento {int(f.get('intento', 0))}: {f.get('accion','')} → {obs}")

    return "\n".join(resumen)
