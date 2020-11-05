""" Port of pscoupe rotations from GMT. """

from math import sin, cos, atan2, acos, degrees, radians

EPSIL = 0.00001  # Small number comparison

class NodalPlane:
    def __init__(self, strike, dip, rake):
        self.strike, self.dip, self.rake = strike, dip, rake

    def __repr__(self):
        return f"NodalPlane(strike={self.strike}, dip={self.dip}, rake={self.rake})"


class Mechanism:
    def __init__(self, np1, np2):
        self.np1, self.np2 = np1, np2

    def __repr__(self):
        return f"Mechanism(np1={self.np1}, np2={self.np2})"

    def plot(self, *args, **kwargs):
        """ Thin wrapper on obspy.imaging.beachball.beachball. """
        from obspy.imaging.beachball import beachball

        fig = beachball(fm=(self.np1.strike, self.np1.dip, self.np1.rake), 
                        *args, **kwargs)
        return fig



def rot_nodal_plane(p_lan: NodalPlane, p_ref: NodalPlane) -> NodalPlane:
    """
    Calcule l'azimut, le pendage, le glissement relatifs d'un
    mecanisme par rapport a un plan de reference p_ref
    defini par son azimut et son pendage.
    On regarde la demi-sphere derriere le plan.
    Les angles sont en degres.

    Genevieve Patau, 8 septembre 1992.
    """

    dfi = p_lan.strike - p_ref.strike

    sd = sin(radians(p_lan.dip))
    cd = cos(radians(p_lan.dip))
    sdfi = sin(radians(dfi))
    cdfi = cos(radians(dfi))
    srd = sin(radians(p_ref.dip))
    crd = cos(radians(p_ref.dip))
    sir = sin(radians(p_lan.rake))
    cor = cos(radians(p_lan.rake))
    cdr = cd * crd + cdfi * sd * srd
    cr = -1 * sd * sdfi
    sr = (sd * crd * cdfi - cd * srd)

    plan_r = NodalPlane(0, 0, 0)

    plan_r.strike = degrees(atan2(sr, cr))
    if cdr < 0.:
        plan_r.strike += 180.0   
    plan_r.strike = plan_r.strike % 360
    plan_r.dip = degrees(acos(abs(cdr)))

    cr = cr * (sir * (cd * crd * cdfi + sd * srd) - cor * crd * sdfi) + sr * (cor * cdfi + sir * cd * sdfi)

    sr = (cor * srd * sdfi + sir * (sd * crd - cd * srd * cdfi))

    plan_r.rake = degrees(atan2(sr, cr))

    if cdr < 0.0:
        plan_r.rake +=  180.0
        if plan_r.rake > 180.0:
            plan_r.rake -= 360.0
    
    return plan_r


def rot_meca(meca: Mechanism, p_ref: NodalPlane) -> NodalPlane:
    """
    Project a mechanism into a plane.

    Adapted from pscoupe from GMT
    """
    mecar = Mechanism(NodalPlane(0, 0, 0), NodalPlane(0, 0, 0))

    if abs(meca.np1.strike - p_ref.strike) < EPSIL and abs (meca.np1.dip - p_ref.dip) < EPSIL:
        mecar.np1.strike = 0.
        mecar.np1.dip = 0.
        mecar.np1.rake = (270. - meca.np1.rake) % 360
    else:
        mecar.np1 = rot_nodal_plane(meca.np1, p_ref)

    if abs(meca.np2.strike - p_ref.strike) < EPSIL and abs(meca.np2.dip - p_ref.dip) < EPSIL:
        mecar.np2.strike = 0.
        mecar.np2.dip = 0.
        mecar.np2.rake = (270. - meca.np2.rake) % 360
    else:
        mecar.np2 = rot_nodal_plane(meca.np2, p_ref)

    if cos(radians(mecar.np2.dip)) < EPSIL and abs(mecar.np1.rake - mecar.np2.rake) < 90.0:
        mecar.np1.strike += 180.0
        mecar.np1.rake += 180.0
        mecar.np1.strike = mecar.np1.strike % 360
        if mecar.np1.rake > 180.0:
            mecar.np1.rake -= 360.0
    return mecar
