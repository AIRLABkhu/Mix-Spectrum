import xml.etree.ElementTree as ET
import os
import numpy as np
from preset_customization import DEFAULT_TEXTURE_ALIAS, DEFAULT_TEXTURE_TYPE
from robosuite.utils.mjcf_utils import (
    ALL_TEXTURES,
    TEXTURES,
    xml_path_completion,
    array_to_string,
)
from typing import List

TEXTURES = {
    "WoodRed": "red-wood.png",
    "WoodGreen": "green-wood.png",
    "WoodBlue": "blue-wood.png",
    "WoodLight": "light-wood.png",
    "WoodDark": "dark-wood.png",
    "WoodTiles": "wood-tiles.png",
    "WoodPanels": "wood-varnished-panels.png",
    "WoodgrainGray": "gray-woodgrain.png",
    "PlasterCream": "cream-plaster.png",
    "PlasterPink": "pink-plaster.png",
    "PlasterYellow": "yellow-plaster.png",
    "PlasterGray": "gray-plaster.png",
    "PlasterWhite": "white-plaster.png",
    "BricksWhite": "white-bricks.png",
    "Metal": "metal.png",
    "SteelBrushed": "steel-brushed.png",
    "SteelScratched": "steel-scratched.png",
    "Brass": "brass-ambra.png",
    "Bread": "bread.png",
    "Can": "can.png",
    "Ceramic": "ceramic.png",
    "Cereal": "cereal.png",
    "Clay": "clay.png",
    "Dirt": "dirt.png",
    "Glass": "glass.png",
    "FeltGray": "gray-felt.png",
    "Lemon": "lemon.png",
    "Custom01": "custom01.png",
    "Custom02": "custom02.png",
    "Custom03": "custom03.png",
    "Custom04": "custom04.png",
    "Custom05": "custom05.png",
    "Custom06": "custom06.png",
    "Custom07": "custom07.png",
    "Custom08": "custom08.png",
    "Custom09": "custom09.png",
    "Custom10": "custom10.png",
    "Custom11": "custom11.png",
    "Custom12": "custom12.png",
    "Custom13": "custom13.png",
    "Custom14": "custom14.png",
    "Custom15": "custom15.png",
    "Custom16": "custom16.png",
    "Custom17": "custom17.png",
    "Custom18": "custom18.png",
    "Custom19": "custom19.png",
    "Custom20": "custom20.png",
    "Custom21": "custom21.png",
    "Custom22": "custom22.png",
    "Custom23": "custom23.png",
    "Custom24": "custom24.png",
    "Custom25": "custom25.png",
    "Custom26": "custom26.png",
    "Custom27": "custom27.png",
    "Custom28": "custom28.png",
    "Custom29": "custom29.png",
    "Custom30": "custom30.png",
    "Custom31": "custom31.png",
    "Custom32": "custom32.png",
    "Custom33": "custom33.png",
    "Custom34": "custom34.png",
    "Custom35": "custom35.png",
    "Custom36": "custom36.png",
    "Custom37": "custom37.png",
    "Custom38": "custom38.png",
    "Custom39": "custom39.png",
    "Custom40": "custom40.png",
}

ALL_TEXTURES = TEXTURES.keys()


class CustomMujocoXML(ET.ElementTree):
    """
    Class to modify the Mujoco XML of Mujoco-enabled env
    """

    def __init__(self, element=None, file=None):
        super(CustomMujocoXML, self).__init__(element=element, file=file)

    @classmethod
    def build_from_element(cls, root):
        return cls(element=root)

    @classmethod
    def build_from_file(cls, xml_fname):
        return cls(file=xml_fname)

    @classmethod
    def build_from_env(cls, env):
        """
        Build from a RobosuiteAdapter env or a robosuite env
        """
        from adapter import RobosuiteAdapter

        if isinstance(env, RobosuiteAdapter):
            env = env.env
        element = env.model.root
        return cls.build_from_element(element)

    def to_string(self):
        return ET.tostring(self.getroot(), encoding="unicode")

    def get_elements_with_tag(self, tag):
        children = []
        for child in self.getroot().iter(tag):
            children.append(child)
        return children

    def get_element_with_name(self, tag, name):
        elems = self.get_elements_with_tag(tag)
        for elem in elems:
            if elem.get("name") == name:
                return elem
        return None

    def remove_element(self, tag, name=None):
        cur = self.getroot()
        CustomMujocoXML._remove_helper(cur, tag, name)

    @staticmethod
    def _remove_helper(cur, tag, name=None):
        for elem in cur:
            if elem.tag == tag:
                if name is not None:
                    if elem.get("name") == name:
                        cur.remove(elem)
                else:
                    cur.remove(elem)
            else:
                CustomMujocoXML._remove_helper(elem, tag, name)

    def get_attributes(self, tag, name, attrib_name):
        elem = self.get_element_with_name(tag=tag, name=name)
        if elem is not None:
            return elem.get(attrib_name)
        else:
            raise ValueError(f"Can't find element with tag {tag} and name {name}")

    def set_attributes(self, tag_name, id_tuple, **kwargs):
        for child in self.getroot().iter(tag_name):
            if child.get(id_tuple[0], default=None) == id_tuple[1]:
                for k, v in kwargs.items():
                    child.set(k, v)

    def add_element_to_parent(self, parent_tag, element):
        """
        Add element as a child of a parent that's identified by its tag
        """
        parent = self.getroot().find(parent_tag)
        parent.append(element)

    @staticmethod
    def save_elements(filename, root_tag, *elems):
        """
        Append a list of elements to a root with a tag and save the tree to a file
        """
        root = ET.Element(root_tag)
        for element in elems:
            root.append(element)
        tree = ET.ElementTree(element=root)
        tree.write(filename)

    def load_elements_from_file(self, filename, name_prefix=""):
        """
        Load elements from an xml file generated by .save_elements(). Update current
        elements with the new elements from file.

        Args:
            filename: file name
            name_prefix: prefix to be attached to every name in the xml as identifier
        """
        new_assets = ET.parse(filename)
        root = new_assets.getroot()
        for child in root.getchildren():
            child.set("name", name_prefix + child.get("name"))
            elem = self.get_element_with_name(child.tag, child.get("name"))
            if elem is not None:
                for k, v in child.items():
                    elem.set(k, v)
            else:
                self.add_element_to_parent(root.tag, child)

    # ----------------------------------------------------
    # ------------ Asset: Texture & Material -------------
    # ----------------------------------------------------
    def set_texture_attributes(self, tex_name, **kwargs):
        self.set_attributes("texture", ("name", tex_name), **kwargs)

    def set_material_attributes(self, mat_name, **kwargs):
        self.set_attributes("material", ("name", mat_name), **kwargs)

    def save_current_tex_mat(self, filename, names_to_save):
        """
        Save the texture/material elements in the xml.
        """
        assets = []
        for tex in self.getroot().iter("texture"):
            if tex.get("name", None) in names_to_save:
                assets.append(tex)

        for mat in self.getroot().iter("material"):
            if mat.get("name", None) in names_to_save:
                assets.append(mat)
        CustomMujocoXML.save_elements(filename, "asset", *assets)

    def get_material_texture(self, mat_name):
        """
        Get the texture file or built-in texture name from a material name
        """
        tex_name = self.get_attributes(
            tag="material", name=mat_name, attrib_name="texture"
        )
        tex_file = self.get_attributes(tag="texture", name=tex_name, attrib_name="file")
        base_name = os.path.basename(os.path.normpath(tex_file))
        for name, file_name in TEXTURES.items():
            if base_name == file_name:
                return name
        return tex_file

    def change_material_texture(self, mat_name, tex_name=None, tex_element=None):
        """
        Change the texture of a material identified by its name. If the given texture
        name is not in the xml file, the given texture element will be added.
        """
        if tex_element is not None:
            self.remove_element("texture", name=tex_element.name)
            self.add_element_to_parent("asset", tex_element)
            self.set_material_attributes(mat_name, texture=tex_element.name)
        elif tex_name is not None:
            textures = self.get_elements_with_tag("texture")
            tex_names = list(map(lambda t: t.get("name", default=None), textures))
            if tex_name not in tex_names:
                if tex_name in ALL_TEXTURES:
                    tex_element = TextureElement.build_from_file(
                        name=tex_name, file_or_texture=tex_name, type="2d"
                    )
                assert tex_element is not None, (
                    "Texture name not found in tree. Must provide a texture element "
                    "to add to the tree."
                )
                self.add_element_to_parent("asset", tex_element)
            self.set_material_attributes(mat_name, texture=tex_name)
        else:
            raise AssertionError(
                "Either a texture name or a texture element should be provided"
            )

    def print_texture_material_info(self):
        """
        Print information about the texture and material in the xml.
        """

        def attrib_to_string(attrib):
            string = ""
            for k, v in attrib.items():
                string += f"{k}={v}, "
            return string[:-2]

        textures = self.get_elements_with_tag("texture")
        materials = self.get_elements_with_tag("material")

        for mat in materials:
            attrib = mat.attrib.copy()
            name = attrib.pop("name")
            tex_name = attrib.pop("texture")
            tex_attrib = None
            for tex in textures.copy():
                if tex.get("name", default=None) == tex_name:
                    tex_attrib = tex.attrib.copy()
                    tex_attrib.pop("name")
                    textures.remove(tex)
                    break
            assert (
                tex_attrib is not None
            ), f"Texture '{tex_name}' does not exist in tree."
            print(f"material: {name} || {attrib_to_string(attrib)}")
            print(f"\ttexture: {tex_name} || {attrib_to_string(tex_attrib)}")

        print("\nOther textures:")
        for tex in textures:
            name = tex.get("name")
            print(f"\ttexture: {name} || {attrib_to_string(tex.attrib)}")

    # ----------------------------------------------------
    # -------------------- Light -------------------------
    # ----------------------------------------------------
    def name_light(self):
        """
        Rename all light elements. This function is needed because in some env light
        elements have no name.
        """
        light_elems = self.get_elements_with_tag("light")
        for i in range(len(light_elems)):
            elem = light_elems[i]
            elem.set("name", f"light{i+1}")

    def remove_all_lights(self):
        self.remove_element("light")

    def set_light_attributes(self, name, **attrib):
        self.set_attributes("light", ("name", name), **attrib)

    def add_light(self, name, **attrib):
        elem = LightElement(name, **attrib)
        self.add_element_to_parent("worldbody", elem)

    def save_current_lights(self, filename, lights_to_save="All"):
        """
        Save the light elements in the xml. Use "All" to save all the elements
        """
        lights = []
        for tex in self.getroot().iter("light"):
            if lights_to_save == "All":
                lights.append(tex)
            elif tex.get("name") in lights_to_save:
                lights.append(tex)
        CustomMujocoXML.save_elements(filename, "worldbody", *lights)

    # ----------------------------------------------------
    # ------------------ Benchmarking --------------------
    # ----------------------------------------------------


class XMLTextureModder:
    def __init__(
        self,
        seed=None,
        tex_candidate=None,
        tex_to_change=None,
        tex_diff_constraint=None,
        tex_type=DEFAULT_TEXTURE_TYPE,
    ):
        """
        Initialize a modder that randomly changes the texture of specified objects.

        Args:
            seed (int): random seed used to randomize these
                modifications without impacting other numpy seeds / randomizations
            tex_candidate (dict): a dictionary that maps alias names of target objects
                (see DEFAULT_TEXTURE_ALIAS) to some texture candidates. Keys should be
                subsets of keys from DEFAULT_TEXTURE_ALIAS'. Values should be a texture
                candidate or a list of texture candidates. If a list of texture
                candidates is provided, a random candidate will be selected and
                applied to the corresponding target object.
                A texture candidate is either a string representing the path to a image
                texture file, or a tuple of RGB values normalized to [0, 1]. Note that
                you can use the names of robosuite's builtin texture files as texture
                candidate. Check `envs.robosuite.ALL_TEXTURES` for all the
                builtin texture names, or `envs.robosuite.TEXTURES` to get a
                dictionary that maps these names to their source files.
            tex_to_change (list): List of object alias names whose texture need to be
                modified. If None, use default texture list in
                DEFAULT_TASK_TEXTURE_LIST[task]
            tex_diff_constraint (list[set]): List of sets, where each set contains some
                texture keys. Each texture key from a set will be assigned a different
                texture from the other keys in the same set. This arg is used to enforce
                texture difference on important objects so that they are more easily
                identified by vision algorithms. If None, no constraint is used.
        """
        self.seed = seed
        self.tex_candidate = tex_candidate if tex_candidate else {}
        self.tex_to_change = tex_to_change if tex_to_change else []
        self.tex_diff_constraint = tex_diff_constraint if tex_diff_constraint else []
        self.tex_type = tex_type

    def random_texture_change(self, mujoco_xml: CustomMujocoXML):
        if self.seed is None:
            random_state = np.random.mtrand._rand
        else:
            random_state = np.random.RandomState(self.seed)

        def change_texture(tex_key, texture):
            tex_type = self.tex_type.get(tex_key, "cube")
            if isinstance(texture, tuple):
                tex_elem = TextureElement.build_from_rgb(
                    f"tex-{tex_key}", rgb1=texture, type=tex_type
                )
            else:
                tex_elem = TextureElement.build_from_file(
                    f"tex-{tex_key}", file_or_texture=texture, type=tex_type
                )
            mat_name = DEFAULT_TEXTURE_ALIAS.get(tex_key, tex_key)
            mujoco_xml.change_material_texture(mat_name=mat_name, tex_element=tex_elem)

        if self.tex_to_change is None:
            tex_to_change = sorted(list(self.tex_candidate.keys()))
        else:
            tex_to_change = sorted(list(self.tex_to_change))

        if self.tex_diff_constraint is not None:
            for c_set in self.tex_diff_constraint:
                constraint = set()
                for key in c_set:
                    if key in tex_to_change:
                        candidate = self.tex_candidate.get(key, None)
                        if candidate is None:
                            continue
                        elif isinstance(candidate, list):
                            new_c = candidate.copy()
                            for c in new_c:
                                if c in constraint:
                                    new_c.remove(c)
                            candidate = new_c[random_state.randint(len(new_c))]
                        change_texture(key, candidate)
                        tex_to_change.remove(key)
                        constraint.add(candidate)
        while tex_to_change:
            key = tex_to_change.pop()
            candidate = self.tex_candidate.get(key, None)
            if candidate is None:
                continue
            elif isinstance(candidate, list):
                candidate = candidate[random_state.randint(len(candidate))]
            change_texture(key, candidate)


class MaterialElement(ET.Element):
    """
    Class for a material element in MujocoXML.
    Reference: http://www.mujoco.org/book/XMLreference.html#material
    """

    def __init__(self, name, **mat_attribs):
        mat_attribs.update(name=name)
        super(MaterialElement, self).__init__("material", attrib=mat_attribs)


class TextureElement(ET.Element):
    """
    Class for a texture element in MujocoXML.
    Reference: http://www.mujoco.org/book/XMLreference.html#texture
    """

    def __init__(self, name, **tex_attribs):
        self.name = name
        tex_attribs.update(name=name)
        super(TextureElement, self).__init__("texture", attrib=tex_attribs)

    @classmethod

    def build_from_file(cls, name, file_or_texture, type="cube", **tex_attribs):
        import random
        print(file_or_texture)

        if file_or_texture in ALL_TEXTURES:
            file = xml_path_completion(
                os.path.join("textures", TEXTURES[file_or_texture])
            )
        elif os.path.isfile(file_or_texture):
            file = os.path.abspath(file_or_texture)
        else:
            raise FileNotFoundError(
                f"{file_or_texture} does not exist as a file name or as a built-in "
                f"texture name"
            )
        return cls(name=name, file=file, type=type, **tex_attribs)

    @classmethod
    def build_from_rgb(
        cls,
        name,
        rgb1,
        rgb2=None,
        builtin="flat",
        type="cube",
        height="100",
        width="100",
        **tex_attribs,
    ):
        if isinstance(rgb1, (List, tuple)):  # RGB array
            assert len(rgb1) == 3, (
                f"Error: Expected rgb array of length 3. Got "
                f"array of length {len(rgb1)} instead."
            )
            assert (
                max(rgb1) <= 1 and min(rgb1) >= 0
            ), f"All RGB vectors should be in the range [0,1]"
            rgb1 = array_to_string(rgb1)

        if rgb2 is None:
            rgb2 = rgb1
        elif isinstance(rgb2, (List, tuple)):
            assert len(rgb2) == 3, (
                f"Error: Expected rgb array of length 3. Got "
                f"array of length {len(rgb2)} instead."
            )
            assert (
                max(rgb2) <= 1 and min(rgb2) >= 0
            ), f"All RGB vectors should be in the range [0,1]"
            rgb2 = array_to_string(rgb2)
        return cls(
            name=name,
            builtin=builtin,
            rgb1=rgb1,
            rgb2=rgb2,
            type=type,
            height=height,
            width=width,
            **tex_attribs,
        )


class LightElement(ET.Element):
    """
    Class for a light element in MujocoXML.
    Reference: http://www.mujoco.org/book/XMLreference.html#light
    """

    def __init__(self, name, **light_attribs):
        light_attribs.update(name=name)
        super(LightElement, self).__init__("light", attrib=light_attribs)
