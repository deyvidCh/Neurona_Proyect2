def construir_respuesta(hits):
    """
    Recibe una lista de dicts (o model_dump de Pydantic) con campos:
    regla, intento, accion, resultado, observacion, paso_a_paso, evidencia, score
    Devuelve un string con: recomendación (intento exitoso) + aprendizajes (fallidos).
    """
    # Intento principal: el primero exitoso; si no hay, usar el primero de la lista
    exitosos = [h for h in hits if (h.get('resultado', '').strip().lower() == 'exitoso')]
    principal = exitosos[0] if exitosos else (hits[0] if hits else None)

    if not principal:
        return "Sin resultados."

    # Resumen recomendado
    paso_o_accion = principal.get('paso_a_paso') or principal.get('accion', '(sin paso_a_paso)')
    resumen = [
        f"✅ Configuración recomendada para **{principal.get('regla','')}** "
        f"(basado en intento exitoso #{int(principal.get('intento', 0))}):",
        paso_o_accion
    ]

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
